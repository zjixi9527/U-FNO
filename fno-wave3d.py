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



# 数据预处理函数
def prepare_dataloader(batch_size=10, start_index=1, end_index=10, data_path='/public/home/hpc221253/pytorch_gpu/3d/data-3d'):
    print(f"Loading files: combined_data{start_index}.h5 to combined_data{end_index - 1}.h5")

    datasrc = []
    data = []
    for file_index in range(start_index, end_index):
        with h5py.File(f'{data_path}/displacement_data{file_index}.h5', 'r') as hf_source:
            for i in range(1, 101):
                src = f'source{i}'
                src_data = np.array(hf_source[src])
                src_data = np.expand_dims(src_data, axis=-1)  # 形状变为 (N, 32, 32, 1)
                src_data = np.tile(src_data, (1, 1, 50))  # 将其复制拓展为 (32, 32, 50)
                src_data = np.expand_dims(src_data, axis=-1)       # 最后拓展为 (32, 32, 50, 1)            
                datasrc.append(src_data)

        with h5py.File(f'{data_path}/displacement_data{file_index}.h5', 'r') as hf_displacement:
            for i in range(1, 101):
                name = f'displacement{i}'
                data.append(np.array(hf_displacement[name][10:60]).reshape(50, 32, 32, 3))

    datasrc = np.array(datasrc, dtype=np.float32)
    data = np.array(data, dtype=np.float32)

    datasrc_tensor = torch.tensor(datasrc).to('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(data).to('cuda' if torch.cuda.is_available() else 'cpu')

    #normalizer_displacement = GaussianNormalizer(data_tensor.to(torch.float32))
    #data_tensor = normalizer_displacement.encode(data_tensor.to(torch.float32))

    normalizer_source = GaussianNormalizer(datasrc_tensor.to(torch.float32))
    datasrc_tensor = normalizer_source.encode(datasrc_tensor.to(torch.float32))

    data_tensor = torch.transpose(data_tensor, 1, 2)
    data_tensor = torch.transpose(data_tensor, 2, 3)

    dataset = TensorDataset(datasrc_tensor, data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=32, y=32, t=50, c=4)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=32, y=32 t=50, c=3)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width)
        # input channel is 4

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2_x = nn.Linear(128, 1)
        self.fc2_y = nn.Linear(128, 1)
        self.fc2_z= nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)

        x_velocity = self.fc2_x(x)  # 三个投影子网络   特征数量变为1
        y_velocity = self.fc2_y(x)
        z_velocity = self.fc2_z(x)
        velocity = torch.cat((x_velocity, y_velocity, z_velocity), dim=-1)
        return  velocity

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)



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

def train_and_evaluate_model(model, train_loader, test_loader, optimizer, scheduler, num_epochs, save_path, plot_path, log_path, test_result_path):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    loss_log = open(os.path.join(log_path, 'loss_log.txt'), 'w')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    loss_fn = LpLoss(size_average=True)

    for epoch in range(num_epochs):
        running_loss = 0.0

        # 训练部分
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
        loss_log.write(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}\n')
        scheduler.step()

        # 测试部分
        model.eval()
        test_loss = 0.0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}')
        loss_log.write(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}\n')

        # 保存模型
        if (epoch + 1) % 100 == 0:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
        
        # 保存结果图像
        if (epoch + 1) % 50 == 0:
            save_outputs_and_targets(all_outputs, all_targets, os.path.join(plot_path, f'results_epoch_{epoch+1}.h5'))

        model.train()

    # 最后保存测试集的真实值和预测值
    save_test_results(model, test_loader, test_result_path)

    loss_log.close()

# 新增的 save_outputs_and_targets 函数
def save_outputs_and_targets(outputs, targets, file_path):
    with h5py.File(file_path, 'w') as f:
        for i in range(len(outputs)):  # 根据样本数量调整
            f.create_dataset(f'outputs{i+1}', data=outputs[i])
            f.create_dataset(f'targets{i+1}', data=targets[i])

# 修改后的 save_test_results 函数
def save_test_results(model, test_loader, file_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    outputs_list = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    # 将 outputs_list 和 targets_list 展开
    outputs_all = np.concatenate(outputs_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)

    # 保存为H5PY文件
    with h5py.File(file_path, 'w') as f:
        for i in range(len(outputs_all)):  # 修改这里以适应不同数量的样本
            f.create_dataset(f'outputs{i+1}', data=outputs_all[i])
            f.create_dataset(f'targets{i+1}', data=targets_all[i])

    print(f'Test results saved to {file_path}')

if __name__ == "__main__":
    # 定义路径
    data_path = '/public/home/hpc221253/pytorch_gpu/3d/data-3d'
    test_result_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/test_results/results.h5'
    checkpoints_adam_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/model/adam'
    plots_adam_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/kz/adam'
    logs_adam_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/log/adam'
    checkpoints_lbfgs_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/model/lbfgs'
    plots_lbfgs_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/kz/lbfgs'
    logs_lbfgs_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/log/lbfgs'

    # 超参数
    batch_size = 10
    learning_rate = 0.005
    num_epochs_adam = 200
    num_epochs_lbfgs =0

    # 数据加载
    train_loader = prepare_dataloader(batch_size=batch_size, start_index=1, end_index=50, data_path=data_path)
    test_loader = prepare_dataloader(batch_size=batch_size, start_index=101, end_index=102, data_path=data_path)

    # 模型初始化

    # 定义傅里叶模式数量，通常与问题相关
    modes1 = 8  # x方向的傅里叶模式数
    modes2 = 8  # y方向的傅里叶模式数
    modes3 = 8  # z方向的傅里叶模式数
    fno_width = 16  # 网络宽度

    # 初始化模型
    model = FNO3d(modes1=modes1, modes2=modes2, modes3=modes3, width=fno_width)

    # Adam 优化器和调度器
    optimizer_adam = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_adam = optim.lr_scheduler.StepLR(optimizer_adam, step_size=50, gamma=0.5)

    # 使用 Adam 训练模型
    print("Training with Adam optimizer")
    train_and_evaluate_model(model, train_loader, test_loader, optimizer_adam, scheduler_adam, num_epochs_adam, save_path=checkpoints_adam_path, plot_path=plots_adam_path, log_path=logs_adam_path, test_result_path=test_result_path)


    # LBFGS 优化器
    #optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=500, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50)

    # 使用 LBFGS 训练模型
    #print("Training with LBFGS optimizer")
    #train_model_lbfgs(model, train_loader, optimizer_lbfgs, num_epochs=num_epochs_lbfgs, save_path=checkpoints_lbfgs_path, plot_path=plots_lbfgs_path, log_path=logs_lbfgs_path)

    print("Program finished successfully.")
