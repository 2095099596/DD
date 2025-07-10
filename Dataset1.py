import os
import torchvision
# import xarray as xr
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import dask.array as da
from itertools import zip_longest
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import dask.array as da
import xarray as xr
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiFileBatchDataset(Dataset):
    def __init__(self, file_paths, variable_name, batch_size,device='cuda',normalize_slt = False,apply_mean=True, mean_dims=(1, 2),
                 replace_nan_with_zero=False, apply_interpolation=False, target_size=(5, 5), qie=False, resample_daily=False,
                 window_size=365,var_name = ' ',slt_std = False,skt_std = False,lai_std = False,time_last = False,normalize_skt=False,
                 normalize_lai = False,file_idx = 0,replace_nan_with_zero_0 = False, max = 0,del_run=False,
                 num = [],ERA5 = False,test = False,GLDAS = False,test_GLDAS20=False):
        """
        :param file_paths: 多个 NetCDF 文件路径列表
        :param variable_name: 要读取的变量名称
        :param batch_size: 批次大小
        :param device: 存储数据的设备（默认是 CUDA）
        :param apply_mean: 是否对数据进行均值处理（默认是 False）
        :param mean_dims: 求均值的维度（默认是 (2, 3)，假设是 height 和 width 维度）
        """
        self.file_paths = file_paths
        self.variable_name = variable_name
        self.batch_size = batch_size
        self.device = device
        self.apply_mean = apply_mean  # 是否计算均值
        self.mean_dims = mean_dims  # 计算均值时使用的维度
        self.replace_nan_with_zero = replace_nan_with_zero  # 是否将 NaN 替换为 0
        self.apply_interpolation = apply_interpolation  # 是否进行插值
        self.target_size = target_size  # 目标尺寸
        self.qie = qie
        self.resample_daily = resample_daily
        self.var_name = var_name
        self.window_size = window_size
        self.slt_std = slt_std
        self.skt_std = skt_std
        self.lai_std = lai_std
        self.time_last = time_last
        self.normalize_skt = normalize_skt
        self.normalize_lai = normalize_lai
        self.normalize_slt = normalize_slt
        self.file_idx = file_idx
        self.max = max
        self.del_run = del_run
        self.replace_nan_with_zero_0 = replace_nan_with_zero_0
        self.num = num
        self.ERA5 = ERA5
        self.test = test
        self.GLDAS = GLDAS
        self.test_GLDAS20 = test_GLDAS20

        # 打开所有 NetCDF 文件，并为每个文件启用分块读取

        def preprocess(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(-60, 75))
            return ds
        def preprocessERA(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.longitude + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(75, -60))
            return ds

        if self.ERA5:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # print(ds)
                ds = preprocessERA(ds)
                self.datasets.append(ds)
        elif self.GLDAS:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # 将经度从 [-180, 180) 转换到 [0, 360)
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
                ds = ds.sortby('lon')
                ds = preprocess(ds)
                self.datasets.append(ds)
        else:
            self.datasets = []
            for f in self.file_paths:

                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # print(ds)
                ds = preprocess(ds)
                self.datasets.append(ds)


        self.time_cumsum = [0]  # 时间索引累积和


        # 计算所有文件中数据的总长度
        if self.resample_daily:
            # 对每个dataset按天重采样求均值
            self.datasets = [ds.resample(time='1D').mean() for ds in self.datasets]
        else:
            self.datasets_daily = None
             #self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)

        if self.del_run:
            new_time_ds = []
            for ds in self.datasets:
                time_var = ds["time"]
                # 判断是否包含闰日（2月29日）
                if any((time_var.dt.month == 2) & (time_var.dt.day == 29)):
                    # 过滤掉2月29日
                    ds = ds.sel(time=~((time_var.dt.month == 2) & (time_var.dt.day == 29)))
                new_time_ds.append(ds)
            self.datasets = new_time_ds

        if self.replace_nan_with_zero:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.fillna(0)
                new_time_ds.append(ds)
            self.datasets = new_time_ds

        if self.time_last:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.isel({'time': slice(None, -1)})
                new_time_ds.append(ds)
            self.datasets = new_time_ds

        self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)
        self.total_chunks = self.total_length // self.window_size  # 总块数（忽略余数）
        self.file_lengths = len(self.datasets)


        for ds in self.datasets:
            long = len(ds[self.variable_name].time)
            self.num.append(long)

    def reset_indices(self):
        """重置 file_idx、max 和 num"""
        self.file_idx = 0
        self.max = 0
        # self.num = []

    def __len__(self):
        print("调用 __len__")
        return self.total_chunks

    def __getitem__(self,key):

        if (key - (self.max//self.window_size)) // (self.num[self.file_idx]//self.window_size) >0:
            self.max = self.max + self.num[self.file_idx] # 累加
            self.file_idx = self.file_idx + 1

        start_idx = key * self.window_size
        end_idx = (key + 1) * self.window_size
        ds = self.datasets[self.file_idx]
        start_in_file = start_idx - self.max
        end_in_file = end_idx - self.max
        data = ds[self.variable_name].isel(time=slice(start_in_file, end_in_file)).values
        data = torch.tensor(data).to(self.device)

        if self.normalize_slt:
            data = data - 273.15

        if self.normalize_lai:
            data = data*(10368/3202)

        if self.normalize_skt:
            data = data - 273.15

        if self.apply_interpolation:
            data = self._apply_interpolation(data)

        if self.replace_nan_with_zero_0:
            data = self._replace_nan_with_zero(data)


        # 是否求均值 目标值
        if self.apply_mean:
            data = torch.mean(data, dim=(1, 2))

        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Warning: targets contain NaN or Inf")


        return data

    def safe_divide(self, data, std):
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        return data / std_safe


    def _apply_mean(self, data):
        """
        对数据进行均值计算，自动忽略0值和非零值混合情况
        :param data: 需要计算均值的数据 (PyTorch Tensor)
        :return: 计算均值后的数据 (自动保持原始维度)
        """
        # 创建有效值掩码 (非零且非NaN的视为有效值)
        valid_mask = (data != 0) & (~torch.isnan(data))

        # 克隆数据并处理特殊值
        processed_data = data.clone()
        processed_data[~valid_mask] = torch.nan  # 将无效位置设为NaN

        # 计算nanmean时会自动忽略NaN值
        mean_result = torch.nanmean(processed_data,
                                    dim=self.mean_dims,
                                    keepdim=False)

        # 处理全零特殊情况（当某维度全为0时nanmean返回nan）
        all_zero_mask = (valid_mask.sum(dim=self.mean_dims) == 0)
        mean_result[all_zero_mask] = 0.0  # 全零位置设为0

        return mean_result

    def _replace_nan_with_zero(self, data):
        mask = torch.isnan(data) | torch.isinf(data)  # 找到 NaN 或 Inf 的位置
        data = torch.where(mask, torch.zeros_like(data), data)
        return data

    def _apply_interpolation(self, data):
        """
        对数据进行双线性插值处理。
        :param data: 需要插值的数据（PyTorch 张量）
        :return: 插值后的数据
        """
        # 检查数据形状是否为 (batch_size, channels, height, width)
        if data.dim() == 3:
            # 如果数据是 3D，假设最后一个维度是空间维度，需要扩展维度为 4D
            data = data.unsqueeze(1)  # (batch_size, channels, 1, height * width)
        # 对数据进行插值操作
        data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
        data = data.squeeze(1)

        return data



class MultiFileBatchDatasetBiao(Dataset):
    def __init__(self, file_paths, variable_name, batch_size,device='cuda',normalize_slt = False,apply_mean=True, mean_dims=(1, 2),
                 replace_nan_with_zero=False, apply_interpolation=False, target_size=(5, 5), qie=False, resample_daily=False,
                 window_size=365,var_name = ' ',slt_std = False,skt_std = False,lai_std = False,time_last = False,normalize_skt=False,
                 normalize_lai = False,file_idx = 0,replace_nan_with_zero_0 = False, max = 0,del_run=False,
                 num = [],ERA5 = False,test = False,GLDAS = False,test_GLDAS20=False):
        """
        :param file_paths: 多个 NetCDF 文件路径列表
        :param variable_name: 要读取的变量名称
        :param batch_size: 批次大小
        :param device: 存储数据的设备（默认是 CUDA）
        :param apply_mean: 是否对数据进行均值处理（默认是 False）
        :param mean_dims: 求均值的维度（默认是 (2, 3)，假设是 height 和 width 维度）
        """


        self.file_paths = file_paths
        self.variable_name = variable_name
        self.batch_size = batch_size
        self.device = device
        self.apply_mean = apply_mean  # 是否计算均值
        self.mean_dims = mean_dims  # 计算均值时使用的维度
        self.replace_nan_with_zero = replace_nan_with_zero  # 是否将 NaN 替换为 0
        self.apply_interpolation = apply_interpolation  # 是否进行插值
        self.target_size = target_size  # 目标尺寸
        self.qie = qie
        self.resample_daily = resample_daily

        self.var_name = var_name
        self.window_size = window_size
        self.slt_std = slt_std
        self.skt_std = skt_std
        self.lai_std = lai_std
        self.time_last = time_last
        self.normalize_skt = normalize_skt
        self.normalize_lai = normalize_lai
        self.normalize_slt = normalize_slt
        self.file_idx = file_idx
        self.max = max
        self.del_run = del_run
        self.replace_nan_with_zero_0 = replace_nan_with_zero_0
        self.num = num
        self.ERA5 = ERA5
        self.test = test
        self.GLDAS = GLDAS
        self.test_GLDAS20 = test_GLDAS20

        # 打开所有 NetCDF 文件，并为每个文件启用分块读取

        def preprocess(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(-60, 75))
            return ds


        def preprocessERA(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.longitude + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(75, -60))
            return ds

        if self.ERA5:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # print(ds)
                ds = preprocessERA(ds)
                self.datasets.append(ds)
        elif self.GLDAS:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # 将经度从 [-180, 180) 转换到 [0, 360)
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
                ds = ds.sortby('lon')
                ds = preprocess(ds)
                self.datasets.append(ds)
        else:
            self.datasets = []
            for f in self.file_paths:

                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # print(ds)
                ds = preprocess(ds)
                self.datasets.append(ds)




        self.time_cumsum = [0]  # 时间索引累积和


        # 计算所有文件中数据的总长度
        if self.resample_daily:
            # 对每个dataset按天重采样求均值
            self.datasets = [ds.resample(time='1D').mean() for ds in self.datasets]
        else:
            self.datasets_daily = None
             #self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)

        if self.del_run:
            new_time_ds = []
            for ds in self.datasets:
                time_var = ds["time"]
                # 判断是否包含闰日（2月29日）
                if any((time_var.dt.month == 2) & (time_var.dt.day == 29)):
                    # 过滤掉2月29日
                    ds = ds.sel(time=~((time_var.dt.month == 2) & (time_var.dt.day == 29)))
                new_time_ds.append(ds)
            self.datasets = new_time_ds




        if self.replace_nan_with_zero:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.fillna(0)
                new_time_ds.append(ds)
            self.datasets = new_time_ds

        if self.time_last:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.isel({'time': slice(None, -1)})
                new_time_ds.append(ds)
            self.datasets = new_time_ds


        self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)
        self.total_chunks = self.total_length // self.window_size  # 总块数（忽略余数）
        self.file_lengths = len(self.datasets)


        for ds in self.datasets:
            long = len(ds[self.variable_name].time)
            self.num.append(long)
        # self.file_idx = 0
        # self.max = 0


    def reset_indices(self):
        """重置 file_idx、max 和 num"""
        self.file_idx = 0
        self.max = 0
        # self.num = []

    def __len__(self):
        print("调用 __len__")
        return self.total_chunks

    def __getitem__(self,key):

        # print(self.file_idx)
        # print(self.max)
        # print(self.num)

        if (key - (self.max//self.window_size)) // (self.num[self.file_idx]//self.window_size) >0:
            # self.win = self.num[self.file_idx] // 365 + self.win  # 165
            self.max = self.max + self.num[self.file_idx] # 累加
            self.file_idx = self.file_idx + 1

        start_idx = key * self.window_size
        end_idx = (key + 1) * self.window_size
        # file_idx = key//365
        # print((self.total_length // self.total_chunks ) * file_idx)

        ds = self.datasets[self.file_idx]
        # print(ds)
        start_in_file = start_idx - self.max
        end_in_file = end_idx - self.max
        # print(start_in_file, end_in_file)
        data = ds[self.variable_name].isel(time=slice(start_in_file, end_in_file)).values
        # print(data.shape)
        # print(self.variable_name)
        # print(f"正在读取文件: {self.file_paths[self.file_idx]}")
        data = torch.tensor(data).to(self.device)

        # 可选择性地进行插值处理
        # print(self.lai_mean_sel.shape)


        # print(self.skt_std_sel.shape)

       
        if self.apply_interpolation:
            data = self._apply_interpolation(data)

        data = data.unsqueeze(1)

        print(data.shape)
        print(self.skt_mean_sel.shape)
        print(self.skt_std_sel.shape)


        if self.normalize_slt:
            data = data - 273.15

        if self.normalize_lai:
            data = data*(10368/3202)



        if self.normalize_skt:
            data = data - 273.15


        # data = data.squeeze(1)





        if self.replace_nan_with_zero_0:
            data = self._replace_nan_with_zero(data)





        # 是否求均值 目标值
        if self.apply_mean:
            data = torch.mean(data, dim=(1, 2))

        # print('4')
        # print(data.shape)
        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Warning: targets contain NaN or Inf")

        # print(data.shape)

        return data

    def safe_divide(self, data, std):
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        return data / std_safe



    def _apply_mean(self, data):
        """
        对数据进行均值计算，自动忽略0值和非零值混合情况
        :param data: 需要计算均值的数据 (PyTorch Tensor)
        :return: 计算均值后的数据 (自动保持原始维度)
        """
        # 创建有效值掩码 (非零且非NaN的视为有效值)
        valid_mask = (data != 0) & (~torch.isnan(data))

        # 克隆数据并处理特殊值
        processed_data = data.clone()
        processed_data[~valid_mask] = torch.nan  # 将无效位置设为NaN

        # 计算nanmean时会自动忽略NaN值
        mean_result = torch.nanmean(processed_data,
                                    dim=self.mean_dims,
                                    keepdim=False)

        # 处理全零特殊情况（当某维度全为0时nanmean返回nan）
        all_zero_mask = (valid_mask.sum(dim=self.mean_dims) == 0)
        mean_result[all_zero_mask] = 0.0  # 全零位置设为0

        return mean_result

    def _replace_nan_with_zero(self, data):
        mask = torch.isnan(data) | torch.isinf(data)  # 找到 NaN 或 Inf 的位置
        data = torch.where(mask, torch.zeros_like(data), data)
        return data

    def _apply_interpolation(self, data):
        """
        对数据进行双线性插值处理。
        :param data: 需要插值的数据（PyTorch 张量）
        :return: 插值后的数据
        """
        # 检查数据形状是否为 (batch_size, channels, height, width)
        if data.dim() == 3:
            # 如果数据是 3D，假设最后一个维度是空间维度，需要扩展维度为 4D
            data = data.unsqueeze(1)  # (batch_size, channels, 1, height * width)
        # 对数据进行插值操作
        data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
        data = data.squeeze(1)

        return data

class MultiFileBatchDatasetGLDAS(Dataset):
    def __init__(self, file_paths, variable_name, batch_size,device='cuda',normalize_slt = False,apply_mean=True, mean_dims=(1, 2),
                 replace_nan_with_zero=False, apply_interpolation=False, target_size=(5, 5), qie=False, resample_daily=False,
                 window_size=365,var_name = ' ',slt_std = False,skt_std = False,lai_std = False,time_last = False,normalize_skt=False,
                 normalize_lai = False,file_idx = 0,replace_nan_with_zero_0 = False, max = 0,del_run=False,
                 num = [],ERA5 = False,test = False,GLDAS = False,test_GLDAS20=False):
        """
        :param file_paths: 多个 NetCDF 文件路径列表
        :param variable_name: 要读取的变量名称
        :param batch_size: 批次大小
        :param device: 存储数据的设备（默认是 CUDA）
        :param apply_mean: 是否对数据进行均值处理（默认是 False）
        :param mean_dims: 求均值的维度（默认是 (2, 3)，假设是 height 和 width 维度）
        """


        self.file_paths = file_paths
        self.variable_name = variable_name
        self.batch_size = batch_size
        self.device = device
        self.apply_mean = apply_mean  # 是否计算均值
        self.mean_dims = mean_dims  # 计算均值时使用的维度
        self.replace_nan_with_zero = replace_nan_with_zero  # 是否将 NaN 替换为 0
        self.apply_interpolation = apply_interpolation  # 是否进行插值
        self.target_size = target_size  # 目标尺寸
        self.qie = qie
        self.resample_daily = resample_daily

        self.var_name = var_name
        self.window_size = window_size
        self.slt_std = slt_std
        self.skt_std = skt_std
        self.lai_std = lai_std
        self.time_last = time_last
        self.normalize_skt = normalize_skt
        self.normalize_lai = normalize_lai
        self.normalize_slt = normalize_slt
        self.file_idx = file_idx
        self.max = max
        self.del_run = del_run
        self.replace_nan_with_zero_0 = replace_nan_with_zero_0
        self.num = num
        self.ERA5 = ERA5
        self.test = test
        self.GLDAS = GLDAS
        self.test_GLDAS20 = test_GLDAS20

        # 打开所有 NetCDF 文件，并为每个文件启用分块读取

        def preprocess(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(-60, 75))
            return ds


        def preprocessERA(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.longitude.min() < 0:
                ds = ds.assign_coords(longitude=((ds.longitude + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(longitude=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(latitude=slice(-60, 75))
            return ds

        if self.ERA5:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # print(ds)
                ds = preprocessERA(ds)
                self.datasets.append(ds)
        elif self.GLDAS:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # 将经度从 [-180, 180) 转换到 [0, 360)
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
                ds = ds.sortby('lon')
                ds = preprocess(ds)
                self.datasets.append(ds)
        else:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                ds = preprocess(ds)
                self.datasets.append(ds)


        # self.std_ = torch.tensor(std_['stl1'].data).to(self.device)
        # self.skt_std_ = torch.tensor(skt_std_['skt'].data).to(self.device)
        # self.lai_std_ = torch.tensor(lai_std_['__xarray_dataarray_variable__'].data).to(self.device)
        # self.skt_std = torch.tensor(skt_std['skt'].data).to(self.device)

        self.lai_mean_sel = torch.tensor(lai_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
        self.skt_mean_sel = torch.tensor(skt_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
        self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)

        self.lai_std_sel = torch.tensor(lai_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
        self.skt_std_sel = torch.tensor(skt_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
        self.Tsl_std_sel = torch.tensor(Tsl_std_sel['__xarray_dataarray_variable__'].values).to(self.device)


        self.time_cumsum = [0]  # 时间索引累积和


        # 计算所有文件中数据的总长度
        if self.resample_daily:
            # 对每个dataset按天重采样求均值
            self.datasets = [ds.resample(time='1D').mean() for ds in self.datasets]
        else:
            self.datasets_daily = None
             #self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)
        # if self.del_run:
        #     new_time_ds = []
        #     for ds in self.datasets:
        #         time_var = ds["time"]
        #         total_time_points = len(time_var)
        #         if total_time_points > 365:
        #             ds = ds.isel({'time': slice(None, -1)})
        #             new_time_ds.append(ds)
        #         else:
        #             new_time_ds.append(ds)
        #         self.datasets = new_time_ds

        if self.del_run:
            new_time_ds = []
            for ds in self.datasets:
                time_var = ds["time"]
                # 判断是否包含闰日（2月29日）
                if any((time_var.dt.month == 2) & (time_var.dt.day == 29)):
                    # 过滤掉2月29日
                    ds = ds.sel(time=~((time_var.dt.month == 2) & (time_var.dt.day == 29)))
                new_time_ds.append(ds)
            self.datasets = new_time_ds




        if self.replace_nan_with_zero:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.fillna(0)
                new_time_ds.append(ds)
            self.datasets = new_time_ds

        if self.time_last:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.isel({'time': slice(None, -1)})
                new_time_ds.append(ds)
            self.datasets = new_time_ds


        self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)
        self.total_chunks = self.total_length // self.window_size  # 总块数（忽略余数）
        self.file_lengths = len(self.datasets)


        for ds in self.datasets:
            long = len(ds[self.variable_name].time)
            self.num.append(long)
        # self.file_idx = 0
        # self.max = 0


    def reset_indices(self):
        """重置 file_idx、max 和 num"""
        self.file_idx = 0
        self.max = 0
        # self.num = []

    def __len__(self):
        print("调用 __len__")
        return self.total_chunks

    def __getitem__(self,key):

        # print(self.file_idx)
        # print(self.max)
        # print(self.num)

        if (key - (self.max//self.window_size)) // (self.num[self.file_idx]//self.window_size) >0:
            # self.win = self.num[self.file_idx] // 365 + self.win  # 165
            self.max = self.max + self.num[self.file_idx] # 累加
            self.file_idx = self.file_idx + 1

        start_idx = key * self.window_size
        end_idx = (key + 1) * self.window_size
        # file_idx = key//365
        # print((self.total_length // self.total_chunks ) * file_idx)

        ds = self.datasets[self.file_idx]
        # print(ds)
        start_in_file = start_idx - self.max
        end_in_file = end_idx - self.max
        # print(start_in_file, end_in_file)
        data = ds[self.variable_name].isel(time=slice(start_in_file, end_in_file)).values
        # print(data.shape)
        # print(self.variable_name)
        # print(f"正在读取文件: {self.file_paths[self.file_idx]}")
        data = torch.tensor(data).to(self.device)
        # 如果需要，将 NaN 替换为 0

        # 可选择性地进行插值处理
        # print(self.lai_mean_sel.shape)


        # print(self.skt_std_sel.shape)

        if self.test:
            mean_path = '/mnt/hdb/LXH/data/JRA55/mean_tsl.nc'
            std_path = '/mnt/hdb/LXH/data/JRA55/std_tas.nc'

            skt_mean_path = '/mnt/hdb/LXH/data/JRA55/mean_tas.nc'
            skt_std_path = '/mnt/hdb/LXH/data/JRA55/std_tsl.nc'

            mean = xr.open_dataset(mean_path)
            std_ = xr.open_dataset(std_path)

            skt_mean = xr.open_dataset(skt_mean_path)
            skt_std_ = xr.open_dataset(skt_std_path)


            lon_min, lon_max = 0, 360
            lat_min, lat_max = -60, 75

            Tsl_mean_sel = mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            Tsl_std_sel = std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_mean_sel = skt_mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_std_sel = skt_std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))



            self.skt_mean_sel = torch.tensor(skt_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)


            self.skt_std_sel = torch.tensor(skt_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_std_sel = torch.tensor(Tsl_std_sel['__xarray_dataarray_variable__'].values).to(self.device)

            self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 147, 288]
            self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)
            self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_mean_sel = self.Tsl_mean_sel.unsqueeze(1)
            self.Tsl_std_sel = self.Tsl_std_sel.unsqueeze(0).unsqueeze(0)


        if self.test_GLDAS20:

            mean_path = '/mnt/hdb/LXH/data/GLDAS/2.0_mean_soiltmp.nc'
            std_path = '/mnt/hdb/LXH/data/GLDAS/2.0_std_soiltmp.nc'

            skt_mean_path = '/mnt/hdb/LXH/data/GLDAS/2.0_mean_tas.nc'
            skt_std_path = '/mnt/hdb/LXH/data/GLDAS/2.0_std_tas.nc'

            mean = xr.open_dataset(mean_path)
            std_ = xr.open_dataset(std_path)

            skt_mean = xr.open_dataset(skt_mean_path)
            skt_std_ = xr.open_dataset(skt_std_path)


            lon_min, lon_max = 0, 360
            lat_min, lat_max = -60, 75

            Tsl_mean_sel = mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            Tsl_std_sel = std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_mean_sel = skt_mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_std_sel = skt_std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))



            self.skt_mean_sel = torch.tensor(skt_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)


            self.skt_std_sel = torch.tensor(skt_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_std_sel = torch.tensor(Tsl_std_sel['__xarray_dataarray_variable__'].values).to(self.device)

            self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 147, 288]
            self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)
            self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_mean_sel = self.Tsl_mean_sel.unsqueeze(1)
            self.Tsl_mean_sel = F.interpolate(self.Tsl_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_std_sel = self.Tsl_std_sel.unsqueeze(0).unsqueeze(0)
            self.Tsl_std_sel = F.interpolate(self.Tsl_std_sel, size=(72, 144), mode='bilinear', align_corners=False)


        # if self.test_GLDAS20:
              # 在第1维插入一个维度

        # print(data.shape)
        # print(self.skt_mean_sel.shape)
        # print(self.skt_std_sel.shape)


        self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 144, 288]
        # 使用双线性插值，缩小为(72, 144)
        self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
        # 去掉channel维度
        self.skt_mean_sel = self.skt_mean_sel.squeeze(1)  # [365, 72, 144]

        self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)  # [1, 1, 144, 288]
        self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
        self.skt_std_sel = self.skt_std_sel.squeeze(0).squeeze(0)  # [72, 144]
        #
        #
        if self.apply_interpolation:
            data = self._apply_interpolation(data)

        #
        # # print('-------------------------------------------------')




        data = data.squeeze(1)

        print(data.shape)
        print(self.skt_mean_sel.shape)
        print(self.skt_std_sel.shape)

        if self.normalize_slt:
            # print('Normalizing SLT')

            data = data - 273.15
            # data = data - self.Tsl_mean_sel
            # data = data / self.Tsl_std_sel


        if self.normalize_lai:
            # print(data[1, 35, :])
            # count_zeros = (data == 0).sum().item()  # 统计等于0的元素个数
            # print("数据中0的个数是:", count_zeros)
            # print('Normalizing LAI')
            data = data*(10368/3202)
            # data = data - self.lai_mean_sel
            # data = data/self.lai_std_sel


        if self.normalize_skt:
            data = data - 273.15
            # data = data - self.skt_mean_sel
            # data = data / self.skt_std_sel

        # data = data.squeeze(1)



        # # data = data.unsqueeze(1)
        # print(data.shape)
        # print(self.skt_mean_sel.shape)
        # print(self.skt_std_sel.shape)


        if self.replace_nan_with_zero_0:
            data = self._replace_nan_with_zero(data)









        # self.lai_std_sel = self._replace_nan_with_zero(self.lai_std_sel)
        # # print(data.shape)
        # # torch.set_printoptions(profile="full")
        #  #print(data[10,:,:])
        # print(torch.mean(self.lai_std_sel, dim=(0, 1)))
        # print('--------------------------')

        # 是否求均值 目标值
        if self.apply_mean:
            data = torch.mean(data, dim=(1, 2))

        # print('4')
        # print(data.shape)
        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Warning: targets contain NaN or Inf")

        # print(data.shape)

        return data

    def safe_divide(self, data, std):
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        return data / std_safe



    def _normalize(self, data):
        """
        改进版归一化函数，实现：
        1. 计算时忽略0和NaN
        2. 归一化后恢复原始零值位置
        3. 提供阈值控制置零能力
        """
        # ==================== 前置处理 ====================
        # 记录原始零值和NaN位置
        zero_mask = (data == 0)
        nan_mask = torch.isnan(data)

        # ==================== 极值计算 ====================
        # 计算有效值范围（忽略0和NaN）
        temp = data.masked_fill(zero_mask | nan_mask, float('inf'))
        matrix_min = temp.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]

        temp = data.masked_fill(zero_mask | nan_mask, float('-inf'))
        matrix_max = temp.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        # ==================== 安全归一化 ====================
        denominator = matrix_max - matrix_min
        denominator = torch.where(denominator == 0, 1.0, denominator)  # 处理零分母
        normalized = (data - matrix_min) / denominator
        # ==================== 后处理置零 ====================
        # 1. 恢复原始零值
        normalized[zero_mask] = 0.0

        # 2. 阈值置零（示例：将小于0.2的值置零）
        # normalized[normalized < 0.2] = 0.0  # 可按需启用

        # 3. 异常值处理（超出[0,1]范围的置零）
        # normalized[(normalized < 0) | (normalized > 1)] = 0.0  # 按需启用
        # ==================== 特殊值保留 ====================
        normalized[nan_mask] = torch.nan  # 保持原始NaN

        return normalized

    def _apply_mean(self, data):
        """
        对数据进行均值计算，自动忽略0值和非零值混合情况
        :param data: 需要计算均值的数据 (PyTorch Tensor)
        :return: 计算均值后的数据 (自动保持原始维度)
        """
        # 创建有效值掩码 (非零且非NaN的视为有效值)
        valid_mask = (data != 0) & (~torch.isnan(data))

        # 克隆数据并处理特殊值
        processed_data = data.clone()
        processed_data[~valid_mask] = torch.nan  # 将无效位置设为NaN

        # 计算nanmean时会自动忽略NaN值
        mean_result = torch.nanmean(processed_data,
                                    dim=self.mean_dims,
                                    keepdim=False)

        # 处理全零特殊情况（当某维度全为0时nanmean返回nan）
        all_zero_mask = (valid_mask.sum(dim=self.mean_dims) == 0)
        mean_result[all_zero_mask] = 0.0  # 全零位置设为0

        return mean_result

    def _replace_nan_with_zero(self, data):
        mask = torch.isnan(data) | torch.isinf(data)  # 找到 NaN 或 Inf 的位置
        data = torch.where(mask, torch.zeros_like(data), data)
        return data

    def _apply_interpolation(self, data):
        """
        对数据进行双线性插值处理。
        :param data: 需要插值的数据（PyTorch 张量）
        :return: 插值后的数据
        """
        # 检查数据形状是否为 (batch_size, channels, height, width)
        if data.dim() == 3:
            # 如果数据是 3D，假设最后一个维度是空间维度，需要扩展维度为 4D
            data = data.unsqueeze(1)  # (batch_size, channels, 1, height * width)
        # 对数据进行插值操作
        data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
        data = data.squeeze(1)

        return data


class MultiFileBatchDatasetERA5(Dataset):
    def __init__(self, file_paths, variable_name, batch_size,device='cuda',normalize_slt = False,apply_mean=True, mean_dims=(1, 2),
                 replace_nan_with_zero=False, apply_interpolation=False, target_size=(5, 5), qie=False, resample_daily=False,
                 window_size=365,var_name = ' ',slt_std = False,skt_std = False,lai_std = False,time_last = False,normalize_skt=False,
                 normalize_lai = False,file_idx = 0,replace_nan_with_zero_0 = False, max = 0,del_run=False,
                 num = [],ERA5 = False,test = False,GLDAS = False,test_GLDAS20=False,ERA5IN=False):
        """
        :param file_paths: 多个 NetCDF 文件路径列表
        :param variable_name: 要读取的变量名称
        :param batch_size: 批次大小
        :param device: 存储数据的设备（默认是 CUDA）
        :param apply_mean: 是否对数据进行均值处理（默认是 False）
        :param mean_dims: 求均值的维度（默认是 (2, 3)，假设是 height 和 width 维度）
        """


        self.file_paths = file_paths
        self.variable_name = variable_name
        self.batch_size = batch_size
        self.device = device
        self.apply_mean = apply_mean  # 是否计算均值
        self.mean_dims = mean_dims  # 计算均值时使用的维度
        self.replace_nan_with_zero = replace_nan_with_zero  # 是否将 NaN 替换为 0
        self.apply_interpolation = apply_interpolation  # 是否进行插值
        self.target_size = target_size  # 目标尺寸
        self.qie = qie
        self.resample_daily = resample_daily

        self.var_name = var_name
        self.window_size = window_size
        self.slt_std = slt_std
        self.skt_std = skt_std
        self.lai_std = lai_std
        self.time_last = time_last
        self.normalize_skt = normalize_skt
        self.normalize_lai = normalize_lai
        self.normalize_slt = normalize_slt
        self.file_idx = file_idx
        self.max = max
        self.del_run = del_run
        self.replace_nan_with_zero_0 = replace_nan_with_zero_0
        self.num = num
        self.ERA5 = ERA5
        self.test = test
        self.GLDAS = GLDAS
        self.test_GLDAS20 = test_GLDAS20
        self.ERA5IN =ERA5IN

        # 打开所有 NetCDF 文件，并为每个文件启用分块读取

        def preprocess(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(-60, 75))
            return ds


        def preprocessERA(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.longitude.min() < 0:
                ds = ds.assign_coords(longitude=((ds.longitude + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(longitude=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(latitude=slice(75, -60))
            return ds

        def preprocessERAIN(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.longitude + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(75, -60))
            return ds


        if self.ERA5:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'valid_time': batch_size})
                # print(ds)
                ds = preprocessERA(ds)
                ds = ds.sortby('latitude', ascending=True)
                # print(ds)
                self.datasets.append(ds)
        elif self.ERA5IN:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # print(ds)
                ds = preprocessERAIN(ds)
                ds = ds.sortby('lat', ascending=True)
                # print(ds)
                self.datasets.append(ds)
        elif self.GLDAS:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # 将经度从 [-180, 180) 转换到 [0, 360)
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
                ds = ds.sortby('lon')
                ds = preprocess(ds)
                self.datasets.append(ds)
        else:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                ds = preprocess(ds)
                self.datasets.append(ds)


        # self.std_ = torch.tensor(std_['stl1'].data).to(self.device)
        # self.skt_std_ = torch.tensor(skt_std_['skt'].data).to(self.device)
        # self.lai_std_ = torch.tensor(lai_std_['__xarray_dataarray_variable__'].data).to(self.device)
        # self.skt_std = torch.tensor(skt_std['skt'].data).to(self.device)

        self.lai_mean_sel = torch.tensor(lai_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
        self.skt_mean_sel = torch.tensor(skt_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
        self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)

        self.lai_std_sel = torch.tensor(lai_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
        self.skt_std_sel = torch.tensor(skt_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
        self.Tsl_std_sel = torch.tensor(Tsl_std_sel['__xarray_dataarray_variable__'].values).to(self.device)


        self.time_cumsum = [0]  # 时间索引累积和


        # 计算所有文件中数据的总长度
        if self.resample_daily:
            # 对每个dataset按天重采样求均值
            self.datasets = [ds.resample(time='1D').mean() for ds in self.datasets]
        else:
            self.datasets_daily = None
             #self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)
        # if self.del_run:
        #     new_time_ds = []
        #     for ds in self.datasets:
        #         time_var = ds["time"]
        #         total_time_points = len(time_var)
        #         if total_time_points > 365:
        #             ds = ds.isel({'time': slice(None, -1)})
        #             new_time_ds.append(ds)
        #         else:
        #             new_time_ds.append(ds)
        #         self.datasets = new_time_ds

        if self.del_run:
            new_time_ds = []
            for ds in self.datasets:
                time_var = ds["valid_time"]
                # 判断是否包含闰日（2月29日）
                if any((time_var.dt.month == 2) & (time_var.dt.day == 29)):
                    # 过滤掉2月29日
                    ds = ds.sel(valid_time=~((time_var.dt.month == 2) & (time_var.dt.day == 29)))
                new_time_ds.append(ds)
            self.datasets = new_time_ds




        if self.replace_nan_with_zero:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.fillna(0)
                new_time_ds.append(ds)
            self.datasets = new_time_ds

        if self.time_last:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.isel({'valid_time': slice(None, -1)})
                new_time_ds.append(ds)
            self.datasets = new_time_ds


        self.total_length = sum(len(ds[self.variable_name].valid_time) for ds in self.datasets)
        self.total_chunks = self.total_length // self.window_size  # 总块数（忽略余数）
        self.file_lengths = len(self.datasets)


        for ds in self.datasets:
            long = len(ds[self.variable_name].valid_time)
            self.num.append(long)
        # self.file_idx = 0
        # self.max = 0


    def reset_indices(self):
        """重置 file_idx、max 和 num"""
        self.file_idx = 0
        self.max = 0
        # self.num = []

    def __len__(self):
        print("调用 __len__")
        return self.total_chunks

    def __getitem__(self,key):

        # print(self.file_idx)
        # print(self.max)
        # print(self.num)

        # print(self.datasets)

        if (key - (self.max//self.window_size)) // (self.num[self.file_idx]//self.window_size) >0:
            # self.win = self.num[self.file_idx] // 365 + self.win  # 165
            self.max = self.max + self.num[self.file_idx] # 累加
            self.file_idx = self.file_idx + 1

        start_idx = key * self.window_size
        end_idx = (key + 1) * self.window_size
        # file_idx = key//365
        # print((self.total_length // self.total_chunks ) * file_idx)

        ds = self.datasets[self.file_idx]
        # print(ds)
        start_in_file = start_idx - self.max
        end_in_file = end_idx - self.max
        # print(start_in_file, end_in_file)
        data = ds[self.variable_name].isel(valid_time=slice(start_in_file, end_in_file)).values
        # print(data.shape)
        # print(self.variable_name)
        # print(f"正在读取文件: {self.file_paths[self.file_idx]}")
        data = torch.tensor(data).to(self.device)
        # 如果需要，将 NaN 替换为 0

        # 可选择性地进行插值处理
        # print(self.lai_mean_sel.shape)


        # print(self.skt_std_sel.shape)



        if self.test:
            mean_path = '/mnt/hdb/LXH/data/JRA55/mean_tsl.nc'
            std_path = '/mnt/hdb/LXH/data/JRA55/std_tas.nc'

            skt_mean_path = '/mnt/hdb/LXH/data/JRA55/mean_tas.nc'
            skt_std_path = '/mnt/hdb/LXH/data/JRA55/std_tsl.nc'

            mean = xr.open_dataset(mean_path)
            std_ = xr.open_dataset(std_path)

            skt_mean = xr.open_dataset(skt_mean_path)
            skt_std_ = xr.open_dataset(skt_std_path)

            print(skt_mean)






            lon_min, lon_max = 0, 360
            lat_min, lat_max = -60, 75

            Tsl_mean_sel = mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            Tsl_std_sel = std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_mean_sel = skt_mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_std_sel = skt_std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))




            self.skt_mean_sel = torch.tensor(skt_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)


            self.skt_std_sel = torch.tensor(skt_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_std_sel = torch.tensor(Tsl_std_sel['__xarray_dataarray_variable__'].values).to(self.device)

            self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 147, 288]
            self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)
            self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_mean_sel = self.Tsl_mean_sel.unsqueeze(1)
            self.Tsl_std_sel = self.Tsl_std_sel.unsqueeze(0).unsqueeze(0)

        def preprocessERA2(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(75, -60))
            return ds

        if self.test_GLDAS20:

            mean_path = '/mnt/hdb/LXH/data/Era5/mean_tsl.nc'
            std_path = '/mnt/hdb/LXH/data/Era5/std_tsl.nc'

            skt_mean_path = '/mnt/hdb/LXH/data/Era5/mean_tas.nc'
            skt_std_path = '/mnt/hdb/LXH/data/Era5/std_tas.nc'

            mean = xr.open_dataset(mean_path)
            std_ = xr.open_dataset(std_path)

            skt_mean = xr.open_dataset(skt_mean_path)
            skt_std_ = xr.open_dataset(skt_std_path)

            # print(skt_mean)

            skt_std_sel = preprocessERA2(skt_std_)
            skt_mean_sel = preprocessERA2(skt_mean)
            Tsl_mean_sel = preprocessERA2(mean)
            Tsl_std_sel = preprocessERA2(std_)

            skt_std_sel = skt_std_sel.sortby('lat', ascending=True)
            skt_mean_sel = skt_mean_sel.sortby('lat', ascending=True)
            Tsl_mean_sel = Tsl_mean_sel.sortby('lat', ascending=True)
            Tsl_std_sel = Tsl_std_sel.sortby('lat', ascending=True)

            # lon_min, lon_max = 0, 360
            # lat_min, lat_max = -60, 75
            #
            # Tsl_mean_sel = mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            # Tsl_std_sel = std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            # skt_mean_sel = skt_mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            # skt_std_sel = skt_std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            # print(skt_mean_sel.shape)

            # slice_0 = inputs1[0, :, :].cpu().numpy()  # 转成numpy数组方便matplotlib使用
            # # 可视化
            # plt.imshow(slice_0, cmap='viridis')  # 你可以换成 'gray' 或其他colormap
            # plt.colorbar()
            # plt.title('Slice 0 visualization')
            # plt.show()
            # #
            # slice_2 = inputs2[0, :, :].cpu().numpy()  # 转成numpy数组方便matplotlib使用
            # # 可视化
            # plt.imshow(slice_2, cmap='viridis')  # 你可以换成 'gray' 或其他colormap
            # plt.colorbar()
            # plt.title('slice_2 visualization')
            # plt.show()

            self.skt_mean_sel = torch.tensor(skt_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)


            self.skt_std_sel = torch.tensor(skt_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_std_sel = torch.tensor(Tsl_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
            # print(self.skt_mean_sel.shape)

            self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 147, 288]
            # print(self.skt_mean_sel.shape)
            self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)
            self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_mean_sel = self.Tsl_mean_sel.unsqueeze(1)
            self.Tsl_mean_sel = F.interpolate(self.Tsl_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_std_sel = self.Tsl_std_sel.unsqueeze(0).unsqueeze(0)
            self.Tsl_std_sel = F.interpolate(self.Tsl_std_sel, size=(72, 144), mode='bilinear', align_corners=False)


        # if self.test_GLDAS20:
              # 在第1维插入一个维度

        # print(data.shape)
        # print(self.skt_mean_sel.shape)
        # print(self.skt_std_sel.shape)


        self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 144, 288]
        # 使用双线性插值，缩小为(72, 144)
        self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
        # 去掉channel维度
        self.skt_mean_sel = self.skt_mean_sel.squeeze(1)  # [365, 72, 144]

        self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)  # [1, 1, 144, 288]
        self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
        self.skt_std_sel = self.skt_std_sel.squeeze(0).squeeze(0)  # [72, 144]
        #
        #

        if self.apply_interpolation:
            data = self._apply_interpolation(data)

            #
            # # print('-------------------------------------------------')

        data = data.squeeze(1)

        print(data.shape)
        # print(self.skt_mean_sel.shape)
        # print(self.Tsl_mean_sel.shape)

        # slice_0 = self.skt_mean_sel[0,0,:, :].cpu().numpy()  # 转成numpy数组方便matplotlib使用
        # # 可视化
        # plt.imshow(slice_0, cmap='viridis')  # 你可以换成 'gray' 或其他colormap
        # plt.colorbar()
        # plt.title('Slice 0 visualization')
        # plt.show()
        # #
        # slice_2 = data[0,0 ,:, :].cpu().numpy()  # 转成numpy数组方便matplotlib使用
        # # 可视化
        # plt.imshow(slice_2, cmap='viridis')  # 你可以换成 'gray' 或其他colormap
        # plt.colorbar()
        # plt.title('slice_2 visualization')
        # plt.show()
        #
        # slice_3 = self.Tsl_std_sel[0, 0, :, :].cpu().numpy()  # 转成numpy数组方便matplotlib使用
        # # 可视化
        # plt.imshow(slice_3, cmap='viridis')  # 你可以换成 'gray' 或其他colormap
        # plt.colorbar()
        # plt.title('slice_2 visualization')
        # plt.show()


        if self.normalize_slt:
            # print('Normalizing SLT')

            data = data - 273.15
            # data = data - self.Tsl_mean_sel
            # data = data / self.Tsl_std_sel

        if self.normalize_lai:
            # print(data[1, 35, :])
            # count_zeros = (data == 0).sum().item()  # 统计等于0的元素个数
            # print("数据中0的个数是:", count_zeros)
            # print('Normalizing LAI')
            data = data * (10368 / 3202)
            # data = data - self.lai_mean_sel
            # data = data/self.lai_std_sel

        if self.normalize_skt:
            data = data - 273.15
            # data = data - self.skt_mean_sel
            # data = data / self.skt_std_sel

        # data = data.squeeze(1)
        self.Tsl_mean_sel = self._replace_nan_with_zero(self.Tsl_mean_sel)
        # print(self.Tsl_mean_sel[0, 0:50, 36])
        # print(data[0, 0:50, 36])
        print(self.Tsl_mean_sel.shape)
        print(data.shape)
        data = self.zero_4d_based_on_a2(self.Tsl_mean_sel, data)
        # print(data[0, 0:50, 36])



        # # data = data.unsqueeze(1)
        # print(data.shape)
        # print(self.skt_mean_sel.shape)
        # print(self.skt_std_sel.shape)


        if self.replace_nan_with_zero_0:
            data = self._replace_nan_with_zero(data)









        # self.lai_std_sel = self._replace_nan_with_zero(self.lai_std_sel)
        # # print(data.shape)
        # # torch.set_printoptions(profile="full")
        #  #print(data[10,:,:])
        # print(torch.mean(self.lai_std_sel, dim=(0, 1)))
        # print('--------------------------')

        # 是否求均值 目标值
        if self.apply_mean:
            data = torch.mean(data, dim=(1, 2))

        # print('4')
        # print(data.shape)
        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Warning: targets contain NaN or Inf")

        # print(data.shape)

        return data

    def zero_4d_based_on_a2(self, a_4d, b_4d):
        """
        四维矩阵操作（例如形状为 [batch, channel, height, width]）
        参数：
            a_4d : ndarray，四维数组
            b_4d : ndarray，四维数组，与a_4d形状相同
        返回：
            修改后的b_4d
        """
        assert a_4d.shape == b_4d.shape, "形状必须一致"
        mask = (a_4d == 0)
        b_4d[mask] = 0
        return b_4d

    def safe_divide(self, data, std):
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        return data / std_safe



    def _normalize(self, data):
        """
        改进版归一化函数，实现：
        1. 计算时忽略0和NaN
        2. 归一化后恢复原始零值位置
        3. 提供阈值控制置零能力
        """
        # ==================== 前置处理 ====================
        # 记录原始零值和NaN位置
        zero_mask = (data == 0)
        nan_mask = torch.isnan(data)

        # ==================== 极值计算 ====================
        # 计算有效值范围（忽略0和NaN）
        temp = data.masked_fill(zero_mask | nan_mask, float('inf'))
        matrix_min = temp.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]

        temp = data.masked_fill(zero_mask | nan_mask, float('-inf'))
        matrix_max = temp.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        # ==================== 安全归一化 ====================
        denominator = matrix_max - matrix_min
        denominator = torch.where(denominator == 0, 1.0, denominator)  # 处理零分母
        normalized = (data - matrix_min) / denominator
        # ==================== 后处理置零 ====================
        # 1. 恢复原始零值
        normalized[zero_mask] = 0.0

        # 2. 阈值置零（示例：将小于0.2的值置零）
        # normalized[normalized < 0.2] = 0.0  # 可按需启用

        # 3. 异常值处理（超出[0,1]范围的置零）
        # normalized[(normalized < 0) | (normalized > 1)] = 0.0  # 按需启用
        # ==================== 特殊值保留 ====================
        normalized[nan_mask] = torch.nan  # 保持原始NaN

        return normalized

    def _apply_mean(self, data):
        """
        对数据进行均值计算，自动忽略0值和非零值混合情况
        :param data: 需要计算均值的数据 (PyTorch Tensor)
        :return: 计算均值后的数据 (自动保持原始维度)
        """
        # 创建有效值掩码 (非零且非NaN的视为有效值)
        valid_mask = (data != 0) & (~torch.isnan(data))

        # 克隆数据并处理特殊值
        processed_data = data.clone()
        processed_data[~valid_mask] = torch.nan  # 将无效位置设为NaN

        # 计算nanmean时会自动忽略NaN值
        mean_result = torch.nanmean(processed_data,
                                    dim=self.mean_dims,
                                    keepdim=False)

        # 处理全零特殊情况（当某维度全为0时nanmean返回nan）
        all_zero_mask = (valid_mask.sum(dim=self.mean_dims) == 0)
        mean_result[all_zero_mask] = 0.0  # 全零位置设为0

        return mean_result

    def _replace_nan_with_zero(self, data):
        mask = torch.isnan(data) | torch.isinf(data)  # 找到 NaN 或 Inf 的位置
        data = torch.where(mask, torch.zeros_like(data), data)
        return data

    def _apply_interpolation(self, data):
        """
        对数据进行双线性插值处理。
        :param data: 需要插值的数据（PyTorch 张量）
        :return: 插值后的数据
        """
        # 检查数据形状是否为 (batch_size, channels, height, width)
        if data.dim() == 3:
            # 如果数据是 3D，假设最后一个维度是空间维度，需要扩展维度为 4D
            data = data.unsqueeze(1)  # (batch_size, channels, 1, height * width)
        # 对数据进行插值操作
        data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
        data = data.squeeze(1)

        return data



class MultiFileBatchDatasetERA5IN(Dataset):
    def __init__(self, file_paths, variable_name, batch_size,device='cuda',normalize_slt = False,apply_mean=True, mean_dims=(1, 2),
                 replace_nan_with_zero=False, apply_interpolation=False, target_size=(5, 5), qie=False, resample_daily=False,
                 window_size=365,var_name = ' ',slt_std = False,skt_std = False,lai_std = False,time_last = False,normalize_skt=False,
                 normalize_lai = False,file_idx = 0,replace_nan_with_zero_0 = False, max = 0,del_run=False,
                 num = [],ERA5 = False,test = False,GLDAS = False,test_GLDAS20=False,ERA5IN=False):
        """
        :param file_paths: 多个 NetCDF 文件路径列表
        :param variable_name: 要读取的变量名称
        :param batch_size: 批次大小
        :param device: 存储数据的设备（默认是 CUDA）
        :param apply_mean: 是否对数据进行均值处理（默认是 False）
        :param mean_dims: 求均值的维度（默认是 (2, 3)，假设是 height 和 width 维度）
        """


        self.file_paths = file_paths
        self.variable_name = variable_name
        self.batch_size = batch_size
        self.device = device
        self.apply_mean = apply_mean  # 是否计算均值
        self.mean_dims = mean_dims  # 计算均值时使用的维度
        self.replace_nan_with_zero = replace_nan_with_zero  # 是否将 NaN 替换为 0
        self.apply_interpolation = apply_interpolation  # 是否进行插值
        self.target_size = target_size  # 目标尺寸
        self.qie = qie
        self.resample_daily = resample_daily

        self.var_name = var_name
        self.window_size = window_size
        self.slt_std = slt_std
        self.skt_std = skt_std
        self.lai_std = lai_std
        self.time_last = time_last
        self.normalize_skt = normalize_skt
        self.normalize_lai = normalize_lai
        self.normalize_slt = normalize_slt
        self.file_idx = file_idx
        self.max = max
        self.del_run = del_run
        self.replace_nan_with_zero_0 = replace_nan_with_zero_0
        self.num = num
        self.ERA5 = ERA5
        self.test = test
        self.GLDAS = GLDAS
        self.test_GLDAS20 = test_GLDAS20
        self.ERA5IN =ERA5IN

        # 打开所有 NetCDF 文件，并为每个文件启用分块读取

        def preprocess(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(-60, 75))
            return ds


        def preprocessERA(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.longitude.min() < 0:
                ds = ds.assign_coords(longitude=((ds.longitude + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(longitude=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(latitude=slice(75, -60))
            return ds

        def preprocessERAIN(ds):
            # 如果经度范围是-180~180，转换为0~360
            if ds.lon.min() < 0:
                ds = ds.assign_coords(lon=((ds.longitude + 360) % 360))
            # 选择经度范围-20~50对应的0~360范围，即340~50，跨越0度，需要分段选择
            ds = ds.sel(lon=slice(0, 360))
            # 选择纬度范围-35~17
            ds = ds.sel(lat=slice(75, -60))
            return ds


        if self.ERA5:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'valid_time': batch_size})
                # print(ds)
                ds = preprocessERA(ds)
                ds = ds.sortby('latitude', ascending=True)
                # print(ds)
                self.datasets.append(ds)
        elif self.ERA5IN:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # print(ds)
                ds = preprocessERAIN(ds)
                ds = ds.sortby('lat', ascending=True)
                # print(ds)
                self.datasets.append(ds)
        elif self.GLDAS:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                # 将经度从 [-180, 180) 转换到 [0, 360)
                ds = ds.assign_coords(lon=((ds.lon + 360) % 360))
                ds = ds.sortby('lon')
                ds = preprocess(ds)
                self.datasets.append(ds)
        else:
            self.datasets = []
            for f in self.file_paths:
                ds = xr.open_dataset(f, chunks={'time': batch_size})
                ds = preprocess(ds)
                self.datasets.append(ds)


        # self.std_ = torch.tensor(std_['stl1'].data).to(self.device)
        # self.skt_std_ = torch.tensor(skt_std_['skt'].data).to(self.device)
        # self.lai_std_ = torch.tensor(lai_std_['__xarray_dataarray_variable__'].data).to(self.device)
        # self.skt_std = torch.tensor(skt_std['skt'].data).to(self.device)

        self.lai_mean_sel = torch.tensor(lai_mean_sel['TLAI'].values).to(self.device)
        self.skt_mean_sel = torch.tensor(skt_mean_sel['T850'].values).to(self.device)
        self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['TSOI_10CM'].values).to(self.device)

        self.lai_std_sel = torch.tensor(lai_std_sel['TLAI'].values).to(self.device)
        self.skt_std_sel = torch.tensor(skt_std_sel['T850'].values).to(self.device)
        self.Tsl_std_sel = torch.tensor(Tsl_std_sel['TSOI_10CM'].values).to(self.device)


        self.time_cumsum = [0]  # 时间索引累积和


        # 计算所有文件中数据的总长度
        if self.resample_daily:
            # 对每个dataset按天重采样求均值
            self.datasets = [ds.resample(time='1D').mean() for ds in self.datasets]
        else:
            self.datasets_daily = None
             #self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)
        # if self.del_run:
        #     new_time_ds = []
        #     for ds in self.datasets:
        #         time_var = ds["time"]
        #         total_time_points = len(time_var)
        #         if total_time_points > 365:
        #             ds = ds.isel({'time': slice(None, -1)})
        #             new_time_ds.append(ds)
        #         else:
        #             new_time_ds.append(ds)
        #         self.datasets = new_time_ds

        if self.del_run:
            new_time_ds = []
            for ds in self.datasets:
                time_var = ds["time"]
                # 判断是否包含闰日（2月29日）
                if any((time_var.dt.month == 2) & (time_var.dt.day == 29)):
                    # 过滤掉2月29日
                    ds = ds.sel(time=~((time_var.dt.month == 2) & (time_var.dt.day == 29)))
                new_time_ds.append(ds)
            self.datasets = new_time_ds




        if self.replace_nan_with_zero:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.fillna(0)
                new_time_ds.append(ds)
            self.datasets = new_time_ds

        if self.time_last:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.isel({'time': slice(None, -1)})
                new_time_ds.append(ds)
            self.datasets = new_time_ds


        self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)
        self.total_chunks = self.total_length // self.window_size  # 总块数（忽略余数）
        self.file_lengths = len(self.datasets)


        for ds in self.datasets:
            long = len(ds[self.variable_name].time)
            self.num.append(long)
        # self.file_idx = 0
        # self.max = 0


    def reset_indices(self):
        """重置 file_idx、max 和 num"""
        self.file_idx = 0
        self.max = 0
        # self.num = []

    def __len__(self):
        print("调用 __len__")
        return self.total_chunks

    def __getitem__(self,key):

        # print(self.file_idx)
        # print(self.max)
        # print(self.num)

        # print(self.datasets)

        if (key - (self.max//self.window_size)) // (self.num[self.file_idx]//self.window_size) >0:
            # self.win = self.num[self.file_idx] // 365 + self.win  # 165
            self.max = self.max + self.num[self.file_idx] # 累加
            self.file_idx = self.file_idx + 1

        start_idx = key * self.window_size
        end_idx = (key + 1) * self.window_size
        # file_idx = key//365
        # print((self.total_length // self.total_chunks ) * file_idx)

        ds = self.datasets[self.file_idx]
        print(ds)
        start_in_file = start_idx - self.max
        end_in_file = end_idx - self.max
        # print(start_in_file, end_in_file)
        data = ds[self.variable_name].isel(time=slice(start_in_file, end_in_file)).values
        # print(data.shape)
        # print(self.variable_name)
        # print(f"正在读取文件: {self.file_paths[self.file_idx]}")
        data = torch.tensor(data).to(self.device)
        # 如果需要，将 NaN 替换为 0

        # 可选择性地进行插值处理
        # print(self.lai_mean_sel.shape)


        # print(self.skt_std_sel.shape)

        if self.test:
            mean_path = '/mnt/hdb/LXH/data/JRA55/mean_tsl.nc'
            std_path = '/mnt/hdb/LXH/data/JRA55/std_tas.nc'

            skt_mean_path = '/mnt/hdb/LXH/data/JRA55/mean_tas.nc'
            skt_std_path = '/mnt/hdb/LXH/data/JRA55/std_tsl.nc'

            mean = xr.open_dataset(mean_path)
            std_ = xr.open_dataset(std_path)

            skt_mean = xr.open_dataset(skt_mean_path)
            skt_std_ = xr.open_dataset(skt_std_path)


            lon_min, lon_max = 0, 360
            lat_min, lat_max = -60, 75

            Tsl_mean_sel = mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            Tsl_std_sel = std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_mean_sel = skt_mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_std_sel = skt_std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))



            self.skt_mean_sel = torch.tensor(skt_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)


            self.skt_std_sel = torch.tensor(skt_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_std_sel = torch.tensor(Tsl_std_sel['__xarray_dataarray_variable__'].values).to(self.device)

            self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 147, 288]
            self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)
            self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_mean_sel = self.Tsl_mean_sel.unsqueeze(1)
            self.Tsl_std_sel = self.Tsl_std_sel.unsqueeze(0).unsqueeze(0)


        if self.test_GLDAS20:

            mean_path = '/mnt/hdb/LXH/data/GLDAS/2.0_mean_soiltmp.nc'
            std_path = '/mnt/hdb/LXH/data/GLDAS/2.0_std_soiltmp.nc'

            skt_mean_path = '/mnt/hdb/LXH/data/GLDAS/2.0_mean_tas.nc'
            skt_std_path = '/mnt/hdb/LXH/data/GLDAS/2.0_std_tas.nc'

            mean = xr.open_dataset(mean_path)
            std_ = xr.open_dataset(std_path)

            skt_mean = xr.open_dataset(skt_mean_path)
            skt_std_ = xr.open_dataset(skt_std_path)


            lon_min, lon_max = 0, 360
            lat_min, lat_max = -60, 75

            Tsl_mean_sel = mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            Tsl_std_sel = std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_mean_sel = skt_mean.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            skt_std_sel = skt_std_.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))



            self.skt_mean_sel = torch.tensor(skt_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_mean_sel = torch.tensor(Tsl_mean_sel['__xarray_dataarray_variable__'].values).to(self.device)


            self.skt_std_sel = torch.tensor(skt_std_sel['__xarray_dataarray_variable__'].values).to(self.device)
            self.Tsl_std_sel = torch.tensor(Tsl_std_sel['__xarray_dataarray_variable__'].values).to(self.device)

            self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 147, 288]
            self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)
            self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_mean_sel = self.Tsl_mean_sel.unsqueeze(1)
            self.Tsl_mean_sel = F.interpolate(self.Tsl_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
            self.Tsl_std_sel = self.Tsl_std_sel.unsqueeze(0).unsqueeze(0)
            self.Tsl_std_sel = F.interpolate(self.Tsl_std_sel, size=(72, 144), mode='bilinear', align_corners=False)


        # if self.test_GLDAS20:
              # 在第1维插入一个维度

        # print(data.shape)
        # print(self.skt_mean_sel.shape)
        # print(self.skt_std_sel.shape)


        self.skt_mean_sel = self.skt_mean_sel.unsqueeze(1)  # [365, 1, 144, 288]
        # 使用双线性插值，缩小为(72, 144)
        self.skt_mean_sel = F.interpolate(self.skt_mean_sel, size=(72, 144), mode='bilinear', align_corners=False)
        # 去掉channel维度
        self.skt_mean_sel = self.skt_mean_sel.squeeze(1)  # [365, 72, 144]

        self.skt_std_sel = self.skt_std_sel.unsqueeze(0).unsqueeze(0)  # [1, 1, 144, 288]
        self.skt_std_sel = F.interpolate(self.skt_std_sel, size=(72, 144), mode='bilinear', align_corners=False)
        self.skt_std_sel = self.skt_std_sel.squeeze(0).squeeze(0)  # [72, 144]
        #
        #

        if self.apply_interpolation:
            data = self._apply_interpolation(data)

            #
            # # print('-------------------------------------------------')

        data = data.squeeze(1)

        print(data.shape)
        print(self.skt_mean_sel.shape)
        print(self.Tsl_mean_sel.shape)

        if self.normalize_slt:
            # print('Normalizing SLT')

            data = data - 273.15
            # data = data - self.Tsl_mean_sel
            # data = data / self.Tsl_std_sel

        if self.normalize_lai:
            # print(data[1, 35, :])
            # count_zeros = (data == 0).sum().item()  # 统计等于0的元素个数
            # print("数据中0的个数是:", count_zeros)
            # print('Normalizing LAI')
            data = data * (10368 / 3202)
            # data = data - self.lai_mean_sel
            # data = data/self.lai_std_sel

        if self.normalize_skt:
            data = data - 273.15
            # data = data - self.skt_mean_sel
            # data = data / self.skt_std_sel

        # data = data.squeeze(1)
        self.Tsl_mean_sel = self._replace_nan_with_zero(self.Tsl_mean_sel)
        # print(self.Tsl_mean_sel[0, 0:50, 36])
        # print(data[0, 0:50, 36])
        data = self.zero_4d_based_on_a2(self.Tsl_mean_sel, data)
        # print(data[0, 0:50, 36])



        # # data = data.unsqueeze(1)
        # print(data.shape)
        # print(self.skt_mean_sel.shape)
        # print(self.skt_std_sel.shape)


        if self.replace_nan_with_zero_0:
            data = self._replace_nan_with_zero(data)









        # self.lai_std_sel = self._replace_nan_with_zero(self.lai_std_sel)
        # # print(data.shape)
        # # torch.set_printoptions(profile="full")
        #  #print(data[10,:,:])
        # print(torch.mean(self.lai_std_sel, dim=(0, 1)))
        # print('--------------------------')

        # 是否求均值 目标值
        if self.apply_mean:
            data = torch.mean(data, dim=(1, 2))

        # print('4')
        # print(data.shape)
        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Warning: targets contain NaN or Inf")

        # print(data.shape)

        return data

    def zero_4d_based_on_a2(self, a_4d, b_4d):
        """
        四维矩阵操作（例如形状为 [batch, channel, height, width]）
        参数：
            a_4d : ndarray，四维数组
            b_4d : ndarray，四维数组，与a_4d形状相同
        返回：
            修改后的b_4d
        """
        assert a_4d.shape == b_4d.shape, "形状必须一致"
        mask = (a_4d == 0)
        b_4d[mask] = 0
        return b_4d

    def safe_divide(self, data, std):
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        return data / std_safe



    def _normalize(self, data):
        """
        改进版归一化函数，实现：
        1. 计算时忽略0和NaN
        2. 归一化后恢复原始零值位置
        3. 提供阈值控制置零能力
        """
        # ==================== 前置处理 ====================
        # 记录原始零值和NaN位置
        zero_mask = (data == 0)
        nan_mask = torch.isnan(data)

        # ==================== 极值计算 ====================
        # 计算有效值范围（忽略0和NaN）
        temp = data.masked_fill(zero_mask | nan_mask, float('inf'))
        matrix_min = temp.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]

        temp = data.masked_fill(zero_mask | nan_mask, float('-inf'))
        matrix_max = temp.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        # ==================== 安全归一化 ====================
        denominator = matrix_max - matrix_min
        denominator = torch.where(denominator == 0, 1.0, denominator)  # 处理零分母
        normalized = (data - matrix_min) / denominator
        # ==================== 后处理置零 ====================
        # 1. 恢复原始零值
        normalized[zero_mask] = 0.0

        # 2. 阈值置零（示例：将小于0.2的值置零）
        # normalized[normalized < 0.2] = 0.0  # 可按需启用

        # 3. 异常值处理（超出[0,1]范围的置零）
        # normalized[(normalized < 0) | (normalized > 1)] = 0.0  # 按需启用
        # ==================== 特殊值保留 ====================
        normalized[nan_mask] = torch.nan  # 保持原始NaN

        return normalized

    def _apply_mean(self, data):
        """
        对数据进行均值计算，自动忽略0值和非零值混合情况
        :param data: 需要计算均值的数据 (PyTorch Tensor)
        :return: 计算均值后的数据 (自动保持原始维度)
        """
        # 创建有效值掩码 (非零且非NaN的视为有效值)
        valid_mask = (data != 0) & (~torch.isnan(data))

        # 克隆数据并处理特殊值
        processed_data = data.clone()
        processed_data[~valid_mask] = torch.nan  # 将无效位置设为NaN

        # 计算nanmean时会自动忽略NaN值
        mean_result = torch.nanmean(processed_data,
                                    dim=self.mean_dims,
                                    keepdim=False)

        # 处理全零特殊情况（当某维度全为0时nanmean返回nan）
        all_zero_mask = (valid_mask.sum(dim=self.mean_dims) == 0)
        mean_result[all_zero_mask] = 0.0  # 全零位置设为0

        return mean_result

    def _replace_nan_with_zero(self, data):
        mask = torch.isnan(data) | torch.isinf(data)  # 找到 NaN 或 Inf 的位置
        data = torch.where(mask, torch.zeros_like(data), data)
        return data

    def _apply_interpolation(self, data):
        """
        对数据进行双线性插值处理。
        :param data: 需要插值的数据（PyTorch 张量）
        :return: 插值后的数据
        """
        # 检查数据形状是否为 (batch_size, channels, height, width)
        if data.dim() == 3:
            # 如果数据是 3D，假设最后一个维度是空间维度，需要扩展维度为 4D
            data = data.unsqueeze(1)  # (batch_size, channels, 1, height * width)
        # 对数据进行插值操作
        data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
        data = data.squeeze(1)

        return data


class MultiFileBatchDatasetERA5_(Dataset):
    def __init__(self, file_paths, variable_name, batch_size,device='cuda',normalize_slt = False,apply_mean=True, mean_dims=(1, 2),
                 replace_nan_with_zero=False, apply_interpolation=False, target_size=(5, 5), qie=False, resample_daily=False,
                 window_size=365,var_name = ' ',slt_std = False,skt_std = False,lai_std = False,time_last = False,normalize_skt=False,
                 normalize_lai = False,file_idx = 0,replace_nan_with_zero_0 = False, max = 0,del_run=False,
                 num = [],ERA5 = False):
        """
        :param file_paths: 多个 NetCDF 文件路径列表
        :param variable_name: 要读取的变量名称
        :param batch_size: 批次大小
        :param device: 存储数据的设备（默认是 CUDA）
        :param apply_mean: 是否对数据进行均值处理（默认是 False）
        :param mean_dims: 求均值的维度（默认是 (2, 3)，假设是 height 和 width 维度）
        """
        self.file_paths = file_paths
        self.variable_name = variable_name
        self.batch_size = batch_size
        self.device = device
        self.apply_mean = apply_mean  # 是否计算均值
        self.mean_dims = mean_dims  # 计算均值时使用的维度
        self.replace_nan_with_zero = replace_nan_with_zero  # 是否将 NaN 替换为 0
        self.apply_interpolation = apply_interpolation  # 是否进行插值
        self.target_size = target_size  # 目标尺寸
        self.qie = qie
        self.resample_daily = resample_daily

        self.var_name = var_name
        self.window_size = window_size
        self.slt_std = slt_std
        self.skt_std = skt_std
        self.lai_std = lai_std
        self.time_last = time_last
        self.normalize_skt = normalize_skt
        self.normalize_lai = normalize_lai
        self.normalize_slt = normalize_slt
        self.file_idx = file_idx
        self.max = max
        self.del_run = del_run
        self.replace_nan_with_zero_0 = replace_nan_with_zero_0
        self.num = num
        self.ERA5 = ERA5

        # 打开所有 NetCDF 文件，并为每个文件启用分块读取
        if  self.ERA5:
            self.datasets = [xr.open_dataset(self.file_paths, chunks={'valid_time': 1000})]
            # print(self.datasets)
            target_variable = self.variable_name  # 替换为你想要的特征值名称
            self.datasets = [
                ds.sel(valid_time=slice('1985-01-01', '1995-12-31'))[target_variable]
                for ds in self.datasets
            ]


        else:
            self.datasets = [xr.open_dataset(f, chunks={'valid_time': 1000}) for f in self.file_paths]



        self.std_ = torch.tensor(std_['stl1'].data).to(self.device)
        self.skt_std_ = torch.tensor(skt_std_['skt'].data).to(self.device)
        self.lai_std_ = torch.tensor(lai_std_['__xarray_dataarray_variable__'].data).to(self.device)
        # self.skt_std = torch.tensor(skt_std['skt'].data).to(self.device)


        self.time_cumsum = [0]  # 时间索引累积和


        # 计算所有文件中数据的总长度
        if self.resample_daily:
            # 对每个dataset按天重采样求均值
            self.datasets = [ds.resample(valid_time='1D').mean() for ds in self.datasets]
        else:
            self.datasets_daily = None
             #self.total_length = sum(len(ds[self.variable_name].time) for ds in self.datasets)


        if self.del_run:
            new_time_ds = []
            for ds in self.datasets:
                # 识别闰日
                # print('12')
                time = ds['valid_time']
                # 如果时间变量不是 datetime 类型，尝试解码
                if not np.issubdtype(time.dtype, np.datetime64):
                    time = xr.decode_cf(time)
                is_leap_day = (time.dt.month == 2) & (time.dt.day == 29)
                # 过滤掉闰日
                filtered_ds = ds.where(~is_leap_day, drop=True)
                # chunked_ds = filtered_ds.chunk({'valid_time': 365})
                new_time_ds.append(filtered_ds)
            self.datasets = new_time_ds

        # print(self.datasets)



        if self.replace_nan_with_zero:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.fillna(0)
                new_time_ds.append(ds)
            self.datasets = new_time_ds


        if self.time_last:
            new_time_ds = []
            for ds in self.datasets:
                ds = ds.isel({'valid_time': slice(None, -1)})
                new_time_ds.append(ds)
            self.datasets = new_time_ds

        if self.normalize_slt:
            new_datasets = []
            for ds in self.datasets:
                # print('2341234')
                ds["dayofyear"] = ds["valid_time"].dt.dayofyear
                ds = ds[(ds["dayofyear"] >= 1) & (ds["dayofyear"] <= 365)]
                # print(ds[self.var_name][0, 24, :].values)
                # print(ds)
                # print(ds.valid_time)
                result = ds - mean["stl1"].sel(dayofyear=ds["dayofyear"])
                # print(result[self.variable_name][0, 24, :].values)
                new_datasets.append(result)
            self.datasets = new_datasets


        if self.normalize_lai:
            new_datasets = []
            for ds in self.datasets:
                min_time = ds[self.variable_name].min(dim='valid_time')
                max_time = ds[self.variable_name].max(dim='valid_time')
                var_norm = (ds[self.variable_name] - min_time) / (max_time - min_time)
                ds_norm = ds.copy()
                ds_norm[self.variable_name] = var_norm
                new_datasets.append(ds_norm)
            self.datasets = new_datasets
            # print(self.datasets)

        if self.normalize_skt:
            new_datasets = []
            for ds in self.datasets:
                ds["dayofyear"] = ds["valid_time"].dt.dayofyear
                ds = ds[(ds["dayofyear"] >= 1) & (ds["dayofyear"] <= 365)]
                print(ds)
                result = ds - skt_mean["skt"].sel(dayofyear=ds["dayofyear"])
                # print(result[0, 24, :].values)
                new_datasets.append(result)
            self.datasets = new_datasets


        self.total_length = sum(len(ds.valid_time) for ds in self.datasets)


        self.total_chunks = math.ceil(self.total_length / self.window_size)  # 总块数（忽略余数）
        self.file_lengths = len(self.datasets)
        # print(self.total_length,self.total_chunks, self.file_lengths)


        for ds in self.datasets:
            long = len(ds.valid_time)
            self.num.append(long)
        # self.file_idx = 0
        # self.max = 0
    def reset_indices(self):
        """重置 file_idx、max 和 num"""
        self.file_idx = 0
        self.max = 0
        # self.num = []

    def __len__(self):
        print("调用 __len__")
        return self.total_chunks

    def __getitem__(self,key):

        # print(self.file_idx)
        # print(self.max)
        print(self.datasets)
        print(self.num)

        if (key - (self.max//self.window_size)) // (self.num[self.file_idx]//self.window_size) >0:
            # self.win = self.num[self.file_idx] // 365 + self.win  # 165
            self.max = self.max + self.num[self.file_idx] # 累加
            self.file_idx = self.file_idx + 1

        start_idx = key * self.window_size
        end_idx = (key + 1) * self.window_size
        # file_idx = key//365
        # print((self.total_length // self.total_chunks ) * file_idx)

        ds = self.datasets[self.file_idx]
        start_in_file = start_idx - self.max
        end_in_file = end_idx - self.max
        # print(start_in_file, end_in_file)
        data = ds.isel(valid_time=slice(start_in_file, end_in_file)).values
        # print(data[0,20,:])
        # print(self.variable_name)
        # print(f"正在读取文件: {self.file_paths[self.file_idx]}")
        data = torch.tensor(data).to(self.device)
        # 如果需要，将 NaN 替换为 0
        if self.replace_nan_with_zero_0:
            data = self._replace_nan_with_zero(data)
        # 可选择性地进行插值处理
        if self.apply_interpolation:
            data = self._apply_interpolation(data)

        if self.lai_std:
            data = data
        # print('1')


        if self.slt_std:
            data = self.safe_divide(data, self.std_)
        if self.skt_std:
            data = self.safe_divide(data, self.skt_std_)

        if self.qie:
            data = data[:, 60:65, 35:40]  # 假设数据形状为 (1, 96, 144)，根据您的需要调整
        # 是否求均值 目标值
        if self.apply_mean:
            data = torch.mean(data, dim=(1, 2))

        # print('4')
        #rint(data.shape)
        # time_index = ds.valid_time.isel(valid_time=slice(start_in_file, end_in_file)).values

        return data

    def safe_divide(self, data, std):
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        return data / std_safe

    def _normalize(self, data):
        """
        改进版归一化函数，实现：
        1. 计算时忽略0和NaN
        2. 归一化后恢复原始零值位置
        3. 提供阈值控制置零能力
        """
        # ==================== 前置处理 ====================
        # 记录原始零值和NaN位置
        zero_mask = (data == 0)
        nan_mask = torch.isnan(data)

        # ==================== 极值计算 ====================
        # 计算有效值范围（忽略0和NaN）
        temp = data.masked_fill(zero_mask | nan_mask, float('inf'))
        matrix_min = temp.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]

        temp = data.masked_fill(zero_mask | nan_mask, float('-inf'))
        matrix_max = temp.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        # ==================== 安全归一化 ====================
        denominator = matrix_max - matrix_min
        denominator = torch.where(denominator == 0, 1.0, denominator)  # 处理零分母
        normalized = (data - matrix_min) / denominator
        # ==================== 后处理置零 ====================
        # 1. 恢复原始零值
        normalized[zero_mask] = 0.0

        # 2. 阈值置零（示例：将小于0.2的值置零）
        # normalized[normalized < 0.2] = 0.0  # 可按需启用

        # 3. 异常值处理（超出[0,1]范围的置零）
        # normalized[(normalized < 0) | (normalized > 1)] = 0.0  # 按需启用
        # ==================== 特殊值保留 ====================
        normalized[nan_mask] = torch.nan  # 保持原始NaN

        return normalized

    def _apply_mean(self, data):
        """
        对数据进行均值计算，自动忽略0值和非零值混合情况
        :param data: 需要计算均值的数据 (PyTorch Tensor)
        :return: 计算均值后的数据 (自动保持原始维度)
        """
        # 创建有效值掩码 (非零且非NaN的视为有效值)
        valid_mask = (data != 0) & (~torch.isnan(data))

        # 克隆数据并处理特殊值
        processed_data = data.clone()
        processed_data[~valid_mask] = torch.nan  # 将无效位置设为NaN

        # 计算nanmean时会自动忽略NaN值
        mean_result = torch.nanmean(processed_data,
                                    dim=self.mean_dims,
                                    keepdim=False)

        # 处理全零特殊情况（当某维度全为0时nanmean返回nan）
        all_zero_mask = (valid_mask.sum(dim=self.mean_dims) == 0)
        mean_result[all_zero_mask] = 0.0  # 全零位置设为0

        return mean_result

    def _replace_nan_with_zero(self, data):
        data = torch.where(torch.isnan(data), torch.zeros_like(data), data)

        return data

    def _apply_interpolation(self, data):
        """
        对数据进行双线性插值处理。
        :param data: 需要插值的数据（PyTorch 张量）
        :return: 插值后的数据
        """
        # 检查数据形状是否为 (batch_size, channels, height, width)
        if data.dim() == 3:
            # 如果数据是 3D，假设最后一个维度是空间维度，需要扩展维度为 4D
            data = data.unsqueeze(1)  # (batch_size, channels, 1, height * width)
        # 对数据进行插值操作
        data = F.interpolate(data, size=self.target_size, mode='bilinear', align_corners=False)
        data = data.squeeze(1)

        return data

class CombinedDataset3(Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.length = min(len(dataset1), len(dataset2), len(dataset3))  # 取三个数据集最小长度

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # file_idx = self._get_file_index(idx)
        try:
            sample1 = self.dataset1[idx]
            sample2 = self.dataset2[idx]
            sample3 = self.dataset3[idx]
        except Exception as e:
            print(f"Error in file index {idx}: {e}")
            return None, None, None  # 返回空数据或其他默认值
        return sample1, sample2, sample3


class CombinedDataset2(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        # self.dataset3 = dataset3
        self.length = min(len(dataset1), len(dataset2))  # 取三个数据集最小长度

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample1 = self.dataset1[idx]
        sample2 = self.dataset2[idx]
        # sample3 = self.dataset3[idx]
        return sample1, sample2


class CombinedDataset1(Dataset):
    def __init__(self, dataset1):
        self.dataset1 = dataset1
        # self.dataset3 = dataset3
        self.length = len(dataset1)  # 取三个数据集最小长度

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # file_idx = self._get_file_index(idx)
        try:
            sample1 = self.dataset1[idx]

        except Exception as e:
            print(f"Error in file index {idx}: {e}")
            return None, None, None  # 返回空数据或其他默认值
        return sample1
