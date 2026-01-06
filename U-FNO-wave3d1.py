import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import scipy.io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import math
from timeit import default_timer
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from FNO_2D import GaussianNormalizer, LpLoss
import matplotlib.colors as mcolors
# 数据读取类
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = False
        except:
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = True

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float



def load_data_files(start_index, end_index, data_path):
    datasrc = []
    data = []
    for file_index in range(start_index, end_index):
        with h5py.File(f'{data_path}/displacement_data{file_index}.h5', 'r') as hf_source:
            for i in range(1, 101):
                src = f'source{i}'
                src_data = np.array(hf_source[src])
                src_data = np.expand_dims(src_data, axis=-1)  # 形状变为 (N, 64, 64, 1)
                src_data = np.tile(src_data, (1, 1, 50))  # 将其复制拓展为 (64, 64, 50)
                src_data = np.expand_dims(src_data, axis=-1)  # 最后拓展为 (64, 64, 50, 1)
                datasrc.append(src_data)

        with h5py.File(f'{data_path}/displacement_data{file_index}.h5', 'r') as hf_displacement:
            for i in range(1, 101):
                name = f'displacement{i}'
                data.append(np.array(hf_displacement[name][0:50]).reshape(50, 64, 64, 3))

    datasrc = np.array(datasrc, dtype=np.float32)
    data = np.array(data, dtype=np.float32)

    return datasrc, data

def prepare_dataloader(batch_size=2, data_path='/public/home/hpc221253/pytorch_gpu/3d/data-3d', start_index=1, end_index=6):
    datasrc, data = load_data_files(start_index, end_index, data_path)

    datasrc_tensor = torch.tensor(datasrc, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(data, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 正则化
    normalizer_source = GaussianNormalizer(datasrc_tensor.to(torch.float32))
    datasrc_tensor = normalizer_source.encode(datasrc_tensor.to(torch.float32))

    # 维度转换
    data_tensor = torch.transpose(data_tensor, 1, 2)
    data_tensor = torch.transpose(data_tensor, 2, 3)

    dataset = TensorDataset(datasrc_tensor, data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


###################################################################################
######################################   UNO3d   ##################################


class SpectralConv3d_Uno(nn.Module):
    def __init__(self, in_codim, out_codim,dim1,dim2,dim3, modes1=None, modes2=None, modes3=None):
        super(SpectralConv3d_Uno, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT. 
        dim1 = Default output grid size along x (or 1st dimension of output domain) 
        dim2 = Default output grid size along y ( or 2nd dimension of output domain)
        dim3 = Default output grid size along time t ( or 3rd dimension of output domain)
        Ratio of grid size of the input and output grid size (dim1,dim2,dim3) implecitely 
        set the expansion or contraction farctor along each dimension.
        modes1, modes2, modes3 = Number of fourier modes to consider for the ontegral operator
                                Number of modes must be compatibale with the input grid size 
                                and desired output grid size.
                                i.e., modes1 <= min( dim1/2, input_dim1/2).
                                      modes2 <= min( dim2/2, input_dim2/2)
                                Here input_dim1, input_dim2 are respectively the grid size along 
                                x axis and y axis (or first dimension and second dimension) of the input domain.
                                Other modes also have the same constrain.
        in_codim = Input co-domian dimension
        out_codim = output co-domain dimension  
#   in_codim 和 out_codim：输入和输出的通道数。
#   dim1, dim2, dim3：输出的网格尺寸（分别对应 x, y, t 维度）。
#   modes1, modes2, modes3：使用的傅里叶模式的数量，用于控制频域操作的范围


        """
        in_codim = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels = in_codim
        self.out_channels = out_codim
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        if modes1 is not None:
            self.modes1 = modes1 
            self.modes2 = modes2
            self.modes3 = modes3 
        else:
            self.modes1 = dim1 
            self.modes2 = dim2
            self.modes3 = dim3//2+1

        self.scale = (1 / (2*in_codim))**(1.0/2.0)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.randn(in_codim, out_codim, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):

        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, dim1 = None,dim2=None,dim3=None):
        """
        dim1,dim2,dim3 are the output grid size along (x,y,t)
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        if dim1 is not None:
            self.dim1 = dim1
            self.dim2 = dim2
            self.dim3 = dim3   
        x = x.to(torch.float32)
        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1], norm = 'forward')

        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1, self.dim2, self.dim3//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm = 'forward')
        return x

class pointwise_op_3D(nn.Module):
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3):
        super(pointwise_op_3D,self).__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        dim1,dim2,dim3 are the output dimensions (x,y,t)
        """
        if dim1 is None:
            dim1 = self.dim1
            dim2 = self.dim2
            dim3 = self.dim3
        x_out = self.conv(x)

        ft = torch.fft.rfftn(x_out,dim=[-3,-2,-1])
        ft_u = torch.zeros_like(ft)
        ft_u[:, :, :(dim1//2), :(dim2//2), :(dim3//2)] = ft[:, :, :(dim1//2), :(dim2//2), :(dim3//2)]
        ft_u[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)] = ft[:, :, -(dim1//2):, :(dim2//2), :(dim3//2)]
        ft_u[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)] = ft[:, :, :(dim1//2), -(dim2//2):, :(dim3//2)]
        ft_u[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)] = ft[:, :, -(dim1//2):, -(dim2//2):, :(dim3//2)]
        
        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))

        x_out = torch.nn.functional.interpolate(x_out, size = (dim1, dim2,dim3),mode = 'trilinear',align_corners=True)
        return x_out

class OperatorBlock_3D(nn.Module):
    """
    Normalize = if true performs InstanceNorm3d on the output.
    Non_Lin = if true, applies point wise nonlinearity.
    All other variables are consistent with the SpectralConv3d_Uno class.
    """
    def __init__(self, in_codim, out_codim,dim1, dim2,dim3,modes1,modes2,modes3, Normalize = False,Non_Lin = True):
        super(OperatorBlock_3D,self).__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1,dim2,dim3,modes1,modes2,modes3)
        self.w = pointwise_op_3D(in_codim, out_codim, dim1,dim2,dim3)
        self.normalize = Normalize
        self.non_lin = Non_Lin
        if Normalize:
            self.normalize_layer = torch.nn.InstanceNorm3d(int(out_codim),affine=True)


    def forward(self,x, dim1 = None, dim2 = None, dim3 = None):
        """
        input shape = (batch, in_codim, input_dim1, input_dim2, input_dim3)
        output shape = (batch, out_codim, dim1,dim2,dim3)
        """
        x1_out = self.conv(x,dim1,dim2,dim3)
        x2_out = self.w(x,dim1,dim2,dim3)
        x_out = x1_out + x2_out
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out

class Uno3D_T10(nn.Module):
    """
    The overall network. It contains 4 layers of the Fourier layer.
    1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
    2. 4 layers of the integral operators u' = (W + K)(u).
        W defined by self.w; K defined by self.conv .
    3. Project from the channel space to the output space by self.fc1 and self.fc2 .
    
    input: the solution of the first 50 timesteps (u(1), ..., u(50)).
    input shape: (batchsize, x=S, y=S, t=T, c=1)
    output: the solution of the next 10 timesteps
    output shape: (batchsize, x=S, y=S, t=T, c=1)
    
    S,S,T = grid size along x,y and t (input function)
    S,S,T = grid size along x,y and t (output function)
    
    in_width = 4; [a(x,y,x),x,y,z]
    with = projection dimesion
    pad = padding amount
    pad_both = boolean, if true pad both size of the domain
    factor = scaling factor of the co-domain dimesions 
    """
    def __init__(self, in_width, width,pad = 2, factor = 1, pad_both = False):
        super(Uno3D_T10, self).__init__()

        self.in_width = in_width # input channel
        self.width = width 
        
        self.padding = pad  
        self.pad_both = pad_both

        self.fc = nn.Linear(self.in_width, self.in_width*2)

        self.fc0 = nn.Linear(self.in_width*2, self.width)  
        
        self.conv0 = OperatorBlock_3D(self.width, 2*factor*self.width,72, 72, 50, 36,36, 25, Normalize = True)
        
        self.conv1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 48, 48, 50,  20,20,25)
        
        self.conv2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 36, 36, 50,  16,16,25)
        
        self.conv3 = OperatorBlock_3D(8*factor*self.width, 16*factor*self.width, 18, 18, 50,  6, 6, 25, Normalize = True )
        
        self.conv6 = OperatorBlock_3D(16*factor*self.width, 4*factor*self.width, 36, 36, 50,  16, 16, 25)
        
        self.conv7 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 48, 48, 50,  20, 20, 25, Normalize = True)
        
        self.conv8 = OperatorBlock_3D(4*factor*self.width, 2*self.width, 72, 72, 50,  36, 36, 25) # will be reshaped

        #self.fc1 = nn.Linear(3*self.width, 4*self.width)
        
        self.fc2_x = nn.Linear(3*self.width, 1)
        self.fc2_y = nn.Linear(3*self.width, 1)
        self.fc2_z = nn.Linear(3*self.width, 1)


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x_fc_1 = self.fc(x)
        x_fc_1 = F.gelu(x_fc_1)
        x_fc0 = self.fc0(x_fc_1)

        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        padding = 4  # 填充的大小
        # 对 (64, 64) 这两个维度进行填充，每个维度前后各填充 8 个像素
        x_fc0_padded = F.pad(x_fc0, (0, 0, padding, padding, padding, padding), mode='constant', value=0)
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]   ##分别为xyt

        x_c0 = self.conv0(x_fc0, int(3*D1/4),int(3*D2/4),D3)    ## xy变为四分之三，t不变  特征数量由self.width,变为2*factor*self.width
        x_c1 = self.conv1(x_c0, D1//2, D2//2, D3)                  ## xy变为四分之二，t不变   特征数量由2*self.width,变为4*factor*self.width
        x_c2 = self.conv2(x_c1, D1//4, D2//4, int(1.0*D3))     ## xy变为四分之一，t不变   特征数量由4*self.width,变为8*factor*self.width
        
        x_c3 = self.conv3(x_c2, D1//4, D2//4, int(1.0*D3))    
        
        x_c6 = self.conv6(x_c3,D1//2, D2//2, int(1.0*D3))      ##x_c6和x_c1相同
        x_c6 = torch.cat([x_c6, torch.nn.functional.interpolate(x_c1, size = (x_c6.shape[2], x_c6.shape[3],x_c6.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
                  ##x_c6和x_c1相同，进行叠加，特征数量4*factor*self.width+4*factor*self.width，对应上面conv7的输入为8*factor*self.width

        x_c7 = self.conv7(x_c6, int(3*D1/4),int(3*D2/4),D3)  ##x_c7和x_c0相同，
        x_c7 = torch.cat([x_c7, torch.nn.functional.interpolate(x_c0, size = (x_c7.shape[2], x_c7.shape[3],x_c7.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
                    ##进行叠加，特征数量2*factor*self.width+2*factor*self.width，对应上面conv8的输入为4*factor*self.width

        
        x_c8 = self.conv8(x_c7,D1,D2,D3)    ###特征数量为2*factor*self.width

        x_c8 = torch.cat([x_c8,torch.nn.functional.interpolate(x_fc0, size = (x_c8.shape[2], x_c8.shape[3],x_c8.shape[4]),mode = 'trilinear',align_corners=True)], dim=1)
                ###特征数量为2*factor*self.width+x_fc0的特征数量factor*self.width，一共是3*factor*self.width
        crop_start = 4  # 裁剪的起始点
        crop_end = -4   # 裁剪的终止点

        # 对 (80, 80) 的空间维度进行裁剪，将其还原为 (64, 64)
        x_c8_cropped = x_c8[:, :, crop_start:crop_end, crop_start:crop_end, :]

        # x_c8_cropped 的形状将会是 (batch, 4, 64, 64, 50)

        x_c8 = x_c8.permute(0, 2, 3, 4, 1)

       #x_fc1 = self.fc1(x_c8)       ###特征数量由3*factor*self.width变为4*factor*self.width

        x_fc1 = F.gelu(x_c8)
        x_velocity = self.fc2_x(x_fc1)  # 三个投影子网络   特征数量由4*factor*self.width变为1
        y_velocity = self.fc2_y(x_fc1)
        z_velocity = self.fc2_z(x_fc1)
        velocity = torch.cat((x_velocity, y_velocity, z_velocity), dim=-1)
        #print(f"x_velocity shape after concat: {x_velocity.shape}")
        #print(f"y_velocity shape after concat: {y_velocity.shape}")
        #print(f"z_velocity shape after concat: {z_velocity.shape}")
        #print(f"velocity shape after concat: {velocity .shape}")
        return velocity  # 合并为batch*64*64*50*3
    
    def get_grid(self, shape, device):   ###包含5个特征数量
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 2*np.pi, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 2*np.pi, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((torch.sin(gridx),torch.sin(gridy),torch.cos(gridx),torch.cos(gridy), gridz), dim=-1).to(device)

# 定义L2范数误差
def L2Loss(output, target):
    return torch.norm(output - target, p=2)

class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)

        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        # Normalize data
        rescaled = (result - vmin) / (vmax - vmin)
        rescaled = np.ma.masked_array(rescaled, np.isnan(rescaled))
        mask = np.ma.getmask(rescaled)
        norm = np.where(rescaled < (midpoint - vmin) / (vmax - vmin),
                        0.5 * rescaled / ((midpoint - vmin) / (vmax - vmin)),
                        0.5 + 0.5 * (rescaled - (midpoint - vmin) / (vmax - vmin)) / ((vmax - midpoint) / (vmax - vmin)))

        norm = np.ma.array(norm, mask=mask)
        if is_scalar:
            norm = np.ma.filled(norm, np.nan).item()
        return norm


#scaler = GradScaler()

# 定义训练过程
def train_model(model, optimizer, scheduler, num_epochs, data_path, batch_size=2, files_per_batch=5, save_path='checkpoints_adam', plot_path='plots_adam'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    loss_fn = LpLoss(size_average=True)  # 使用LpLoss损失函数

    total_files = 90
    num_batches = total_files // files_per_batch

    # 加载验证集数据
    validation_loader = prepare_dataloader(batch_size=batch_size, data_path=data_path, start_index=92, end_index=94)

    for epoch in range(num_epochs):
        running_loss = 0.0
        validation_loss = 0.0
        
        # 训练阶段
        for batch_num in range(num_batches):
            start_index = batch_num * files_per_batch + 1
            end_index = start_index + files_per_batch

            train_loader = prepare_dataloader(batch_size=batch_size, data_path=data_path, start_index=start_index, end_index=end_index)
            print(f'load files {start_index} to {end_index}')
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / (len(train_loader.dataset) * num_batches)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            for val_inputs, val_targets in validation_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                val_loss = loss_fn(val_outputs, val_targets)
                validation_loss += val_loss.item() * val_inputs.size(0)
        
        validation_epoch_loss = validation_loss / len(validation_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_epoch_loss:.4f}')
        scheduler.step()
        
        # 保存验证集损失
        if epoch == num_epochs - 1:
            with open(os.path.join(save_path, 'validation_loss.txt'), 'w') as f:
                for val_inputs, val_targets in validation_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = loss_fn(val_outputs, val_targets)
                    f.write(f'{val_loss.item()}')
        # 保存模型
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
        
        # 保存结果图像
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                inputs, targets = next(iter(train_loader))
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 不使用自动混合精度推理
                outputs = model(inputs)
                
                print("outputs shape:", outputs.shape)
                print("targets shape:", targets.shape)

                for t in [9, 19, 29, 39, 49]:  # 每隔10个时间步长
                    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

                    directions = ['X', 'Y', 'Z']
                    for i in range(3):
                        # 真实值
                        vmin_true = targets[0, :, :, t, i].min().item()
                        vmax_true = targets[0, :, :, t, i].max().item()
                        norm_true = MidpointNormalize(vmin=vmin_true, vmax=vmax_true, midpoint=0)

                        im0 = axes[0, i].imshow(targets[0, :, :, t, i].cpu().numpy(), cmap='RdBu_r', norm=norm_true)
                        axes[0, i].set_title(f'True {directions[i]} at time step {t+1}')
                        fig.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)
                        
                        # 预测值
                        vmin_pred = outputs[0, :, :, t, i].min().item()
                        vmax_pred = outputs[0, :, :, t, i].max().item()
                        norm_pred = MidpointNormalize(vmin=vmin_pred, vmax=vmax_pred, midpoint=0)

                        im1 = axes[1, i].imshow(outputs[0, :, :, t, i].cpu().numpy(), cmap='RdBu_r', norm=norm_pred)
                        axes[1, i].set_title(f'Predicted {directions[i]} at time step {t+1}')
                        fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)
                        
                        # 误差
                        error = outputs[0, :, :, t, i] - targets[0, :, :, t, i]
                        vmin_error = error.min().item()
                        vmax_error = error.max().item()
                        norm_error = MidpointNormalize(vmin=vmin_error, vmax=vmax_error, midpoint=0)

                        im2 = axes[2, i].imshow(error.cpu().numpy(), cmap='RdBu_r', norm=norm_error)
                        axes[2, i].set_title(f'Error {directions[i]} at time step {t+1}')
                        fig.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
                    
                    plt.savefig(os.path.join(plot_path, f'result_epoch_{epoch+1}_t{t+1}.png'))
                    plt.close()
            model.train()

#BFGS 优化器的训练
def train_model(model, optimizer, scheduler, num_epochs, data_path, batch_size=2, files_per_batch=5, save_path='checkpoints_adam', plot_path='plots_adam'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    loss_fn = LpLoss(size_average=True)  # 使用LpLoss损失函数

    total_files = 90
    num_batches = total_files // files_per_batch

    # 加载验证集数据
    validation_loader = prepare_dataloader(batch_size=batch_size, data_path=data_path, start_index=92, end_index=94)

    # 打开日志文件记录训练和验证损失
    with open(os.path.join(save_path, 'training_validation_loss.txt'), 'w') as log_file:
        for epoch in range(num_epochs):
            running_loss = 0.0
            validation_loss = 0.0
            
            # 训练阶段
            for batch_num in range(num_batches):
                start_index = batch_num * files_per_batch + 1
                end_index = start_index + files_per_batch

                train_loader = prepare_dataloader(batch_size=batch_size, data_path=data_path, start_index=start_index, end_index=end_index)
                print(f'load files {start_index} to {end_index}')
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / (len(train_loader.dataset) * num_batches)
            
            # 验证阶段
            model.eval()
            with torch.no_grad():
                for val_inputs, val_targets in validation_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = loss_fn(val_outputs, val_targets)
                    validation_loss += val_loss.item() * val_inputs.size(0)
            
            validation_epoch_loss = validation_loss / len(validation_loader.dataset)

            # 输出训练和验证损失
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_epoch_loss:.4f}')
            log_file.write(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation Loss: {validation_epoch_loss:.4f}\n')
            scheduler.step()
            
            # 保存验证集损失
            if epoch == num_epochs - 1:
                with open(os.path.join(save_path, 'final_validation_losses.txt'), 'w') as f:
                    for val_inputs, val_targets in validation_loader:
                        val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                        val_outputs = model(val_inputs)
                        val_loss = loss_fn(val_outputs, val_targets)
                        f.write(f'{val_loss.item()}\n')

        # 保存模型
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
        
        # 保存结果图像
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                inputs, targets = next(iter(train_loader))
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                for t in [9, 19, 29, 39, 49, 59]:  # 每隔10个时间步长
                    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

                    directions = ['X', 'Y', 'Z']
                    for i in range(3):
                        # 真实值
                        vmin_true = targets[0, :, :, t, i].min().item()
                        vmax_true = targets[0, :, :, t, i].max().item()
                        norm_true = MidpointNormalize(vmin=vmin_true, vmax=vmax_true, midpoint=0)

                        im0 = axes[0, i].imshow(targets[0, :, :, t, i].cpu().numpy(), cmap='RdBu_r', norm=norm_true)
                        axes[0, i].set_title(f'True {directions[i]} at time step {t+1}')
                        fig.colorbar(im0, ax=axes[0, i], fraction=0.046, pad=0.04)

                        # 预测值
                        vmin_pred = outputs[0, :, :, t, i].min().item()
                        vmax_pred = outputs[0, :, :, t, i].max().item()
                        norm_pred = MidpointNormalize(vmin=vmin_pred, vmax=vmax_pred, midpoint=0)

                        im1 = axes[1, i].imshow(outputs[0, :, :, t, i].cpu().numpy(), cmap='RdBu_r', norm=norm_pred)
                        axes[1, i].set_title(f'Predicted {directions[i]} at time step {t+1}')
                        fig.colorbar(im1, ax=axes[1, i], fraction=0.046, pad=0.04)

                        # 误差
                        error = outputs[0, :, :, t, i] - targets[0, :, :, t, i]
                        vmin_error = error.min().item()
                        vmax_error = error.max().item()
                        norm_error = MidpointNormalize(vmin=vmin_error, vmax=vmax_error, midpoint=0)

                        im2 = axes[2, i].imshow(error.cpu().numpy(), cmap='RdBu_r', norm=norm_error)
                        axes[2, i].set_title(f'Error {directions[i]} at time step {t+1}')
                        fig.colorbar(im2, ax=axes[2, i], fraction=0.046, pad=0.04)
                    
                    plt.savefig(os.path.join(plot_path, f'result_epoch_{epoch+1}_t{t+1}.png'))
                    plt.close()
            model.train()



if __name__ == "__main__":
    # 定义路径
    data_path = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/data-3d'
    checkpoints_adam_path = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/model/adam'
    plots_adam_path = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/kz/adam'
    logs_adam_path = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/log/adam'
    checkpoints_lbfgs_path = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/model/lbfgs'
    plots_lbfgs_path = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/kz/lbfgs'
    logs_lbfgs_path = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/log/lbfgs'

    # 超参数
    batch_size = 1
    learning_rate = 0.005
    num_epochs_adam = 200
    num_epochs_lbfgs = 100  # 修改为大于0的值进行LBFGS训练

    # 模型初始化
    fno_width = 4  # in_width为输入数据的特征数，width是网络中采用的特征数，其中进行Unet操作变换的就是width
    model = Uno3D_T10(in_width=6, width=fno_width, factor=1)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Adam 优化器和调度器
    optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_adam = optim.lr_scheduler.StepLR(optimizer_adam, step_size=50, gamma=0.5)

    # 使用 Adam 训练模型
    print("Training with Adam optimizer")
    train_model(model, optimizer_adam, scheduler_adam, num_epochs=num_epochs_adam, data_path=data_path, batch_size=batch_size, files_per_batch=5, save_path=checkpoints_adam_path, plot_path=plots_adam_path)

    # LBFGS 优化器
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=500, max_eval=None, tolerance_grad=1e-5, tolerance_change=1e-7, history_size=10)

    # 使用 LBFGS 训练模型
    print("Training with LBFGS optimizer")

    train_model_lbfgs(
        model=model,
        optimizer=optimizer_lbfgs,
        num_epochs=num_epochs_lbfgs,
        data_path=data_path,
        batch_size=batch_size,
        files_per_batch=5,  # 根据之前的设置来分配文件批次
        save_path=checkpoints_lbfgs_path,
        plot_path=plots_lbfgs_path,
        log_path=logs_lbfgs_path
    )


    print("Program finished successfully.")
