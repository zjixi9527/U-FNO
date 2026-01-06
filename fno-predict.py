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

def calculate_rmse(predictions, true_values):
    mse = mean_squared_error(true_values.flatten(), predictions.flatten())
    return np.sqrt(mse)


# 加载训练好的模型
def load_trained_model(model_path, width):
    modes1 = 8  # x方向的傅里叶模式数
    modes2 = 8  # y方向的傅里叶模式数
    modes3 = 8  # z方向的傅里叶模式数
    fno_width = 16  # 网络宽度

    model = FNO3d(modes1=modes1, modes2=modes2, modes3=modes3, width=fno_width)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


# 准备数据加载器
def prepare_dataloader_for_prediction(batch_size=10, start_index=101, end_index=102, data_path='/public/home/hpc221253/pytorch_gpu/3d/data-3d'):
    datasrc = []
    data = []
    for file_index in range(start_index, end_index):
        with h5py.File(f'{data_path}/displacement_data{file_index}.h5', 'r') as hf_source:
            for i in range(1, 101):
                src = f'source{i}'
                src_data = np.array(hf_source[src])
                src_data = np.expand_dims(src_data, axis=-1)  # 形状变为 (N, 32, 32, 1)
                src_data = np.tile(src_data, (1, 1, 50))  # 将其复制拓展为 (32, 32, 50)
                src_data = np.expand_dims(src_data, axis=-1)  # 最后拓展为 (32, 32, 50, 1)
                datasrc.append(src_data)

        with h5py.File(f'{data_path}/displacement_data{file_index}.h5', 'r') as hf_displacement:
            for i in range(1, 101):
                name = f'displacement{i}'
                data.append(np.array(hf_displacement[name][10:60]).reshape(50, 32, 32, 3))

    datasrc = np.array(datasrc, dtype=np.float32)
    data = np.array(data, dtype=np.float32)

    datasrc_tensor = torch.tensor(datasrc).to('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = torch.tensor(data).to('cuda' if torch.cuda.is_available() else 'cpu')

    data_tensor = torch.transpose(data_tensor, 1, 2)
    data_tensor = torch.transpose(data_tensor, 2, 3)

    dataset = TensorDataset(datasrc_tensor, data_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 自己实现均方误差
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


import time

def predict_and_calculate_rmse(model, data_loader, true_data_file, rmse_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    rmse_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            start_time = time.time()  # 开始计时
            
            inputs = inputs.to(device)
            predictions = model(inputs)
            predictions_np = predictions.cpu().numpy()
            targets_np = targets.cpu().numpy()

            # 针对每个批次内的每个算例逐个计算 RMSE
            for i in range(len(predictions_np)):
                rmse = calculate_rmse(predictions_np[i], targets_np[i])
                rmse_list.append(rmse)  # 保存每个算例的 RMSE

                # 逐个保存预测和真实值，并使用 `batch_idx * batch_size + i` 作为 index
                index = batch_idx * data_loader.batch_size + i
                save_predictions_to_h5py(predictions_np[i], targets_np[i], true_data_file, index)

            end_time = time.time()  # 结束计时
            elapsed_time = end_time - start_time  # 计算每次预测的时间
            print(f"Batch {batch_idx + 1} prediction time: {elapsed_time:.4f} seconds")

    # 保存所有 RMSE 结果
    with open(rmse_file, 'w') as f:
        for i, rmse in enumerate(rmse_list):
            f.write(f'{rmse}\n')

    print(f'RMSE for all examples saved to {rmse_file}')




def save_predictions_to_h5py(predictions, true_values, filename, index):
    # 每个算例的结果按index存储
    with h5py.File(filename, 'a') as hf:
        if f'predictions_{index}' in hf:
            del hf[f'predictions_{index}']
        if f'true_values_{index}' in hf:
            del hf[f'true_values_{index}']
        hf.create_dataset(f'predictions_{index}', data=predictions)
        hf.create_dataset(f'true_values_{index}', data=true_values)

def save_rmse_to_txt(rmse, filename, index):
    with open(filename, 'a') as f:  # 使用追加模式写入文件
        f.write(f'Example {index+1} RMSE: {rmse}\n')

if __name__ == "__main__":
    # 定义路径
    model_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/model/adam/model_epoch_200.pth'  # 已训练好的模型路径
    test_result_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/test_results/predictions.h5'  # 保存预测结果
    rmse_result_path = '/public/home/hpc221253/pytorch_gpu/3d/FNO_result/test_results/rmse_results.txt'  # 保存RMSE的路径
    data_path = '/public/home/hpc221253/pytorch_gpu/3d/data-3d'  # 数据路径

    # 超参数
    fno_width = 16  # 模型宽度
    batch_size = 10  # batch size

    # 加载训练好的模型
    model = load_trained_model(model_path, width=fno_width)

    # 准备数据
    test_loader = prepare_dataloader_for_prediction(batch_size=batch_size, start_index=101, end_index=106, data_path=data_path)

    # 进行预测并计算RMSE
    predict_and_calculate_rmse(model, test_loader, test_result_path, rmse_result_path)

    print("Prediction and RMSE calculation completed successfully.")