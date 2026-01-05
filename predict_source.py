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



def load_data_from_single_file(data_path):
    datasrc = []
    # 读取 source1 到 source15 的张量数据
    with h5py.File(data_path, 'r') as hf_source:
        for i in range(1, 101):  # 读取 source1 到 source15
            src = f'source{i}'
            src_data = np.array(hf_source[src])
            src_data = np.expand_dims(src_data, axis=-1)  # 形状变为 (64, 64, 1)
            src_data = np.tile(src_data, (1, 1, 50))  # 将其复制拓展为 (64, 64, 50)
            src_data = np.expand_dims(src_data, axis=-1)  # 最后拓展为 (64, 64, 50, 1)
            datasrc.append(src_data)

    datasrc = np.array(datasrc, dtype=np.float32)
    normalizer_source = GaussianNormalizer(torch.tensor(datasrc, dtype=torch.float32))
    datasrc = normalizer_source.encode(torch.tensor(datasrc, dtype=torch.float32)).numpy()
    return datasrc

def prepare_dataloader(batch_size=2, data_path='sources_data.h5'):
    datasrc = load_data_from_single_file(data_path)

    # 转换为PyTorch Tensor
    datasrc_tensor = torch.tensor(datasrc, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 正则化
    normalizer_source = GaussianNormalizer(datasrc_tensor.to(torch.float32))
    datasrc_tensor = normalizer_source.encode(datasrc_tensor.to(torch.float32))

    dataset = TensorDataset(datasrc_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)



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

from timeit import default_timer
import csv
import os
import numpy as np
import torch
import h5py

def predict(model, dataloader, output_path, warmup=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # 可选：warmup（第一次会触发CUDA kernel/缓存初始化，影响计时）
    if warmup:
        first = next(iter(dataloader), None)
        if first is not None:
            with torch.inference_mode():
                _ = model(first[0].to(device))
            if device == 'cuda':
                torch.cuda.synchronize()

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = default_timer()

    all_predictions = []
    n_samples = 0
    with torch.inference_mode():
        for (inputs,) in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            all_predictions.append(outputs.detach().cpu().numpy())
            n_samples += inputs.shape[0]

    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = default_timer()
    elapsed = t1 - t0

    preds = np.concatenate(all_predictions, axis=0)
    with h5py.File(output_path, 'w') as hf_out:
        hf_out.create_dataset('predictions', data=preds)

    return elapsed, n_samples


def batch_predict(model, input_dir, output_dir, num_files=1, batch_size=2):
    os.makedirs(output_dir, exist_ok=True)

    timing_csv = os.path.join(output_dir, "predict_timing.csv")
    rows = [("file_id", "input_file", "output_file", "n_samples", "time_s", "time_per_sample_s")]

    total_time = 0.0
    total_samples = 0

    for i in range(1, num_files + 1):
        input_file_path = os.path.join(input_dir, f'sources_data_line{i}.h5')
        output_file_path = os.path.join(output_dir, f'predicate{i}.h5')

        dataloader = prepare_dataloader(batch_size=batch_size, data_path=input_file_path)

        print(f"\nPredicting: {input_file_path}")
        elapsed, n_samples = predict(model, dataloader, output_file_path, warmup=True)

        tps = elapsed / max(n_samples, 1)
        print(f"Saved: {output_file_path}")
        print(f"Time: {elapsed:.4f} s | Samples: {n_samples} | {tps:.6f} s/sample")

        rows.append((i, input_file_path, output_file_path, n_samples, f"{elapsed:.6f}", f"{tps:.9f}"))
        total_time += elapsed
        total_samples += n_samples

    # 写CSV
    with open(timing_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print("\n================ Summary ================")
    print(f"Total time: {total_time:.4f} s | Total samples: {total_samples} | "
          f"{(total_time / max(total_samples, 1)):.6f} s/sample")
    print(f"Timing saved to: {timing_csv}")

# 设置路径
input_directory = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/yuceduibi/source'  # 输入 H5 文件目录
output_directory = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/yuceduibi/predicate'  # 输出预测结果目录

# 加载训练好的模型权重
checkpoints_path = '/public/home/hpc221253/pytorch_gpu/3d-menyuan/model/adam/model_epoch_200.pth'

# 创建模型实例
model = Uno3D_T10(in_width=6, width=4, factor=1)

# 加载权重
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.load_state_dict(torch.load(checkpoints_path, map_location=device))

# 循环读取进行预测并保存
batch_predict(model, input_directory, output_directory, num_files=1, batch_size=2)

