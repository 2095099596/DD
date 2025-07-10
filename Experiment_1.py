import os
import torch
import torch.nn as nn
import os
import csv  # 新增
import torch
import numpy as np
import matplotlib.pyplot as plt  # 用于可视化
from datetime import datetime  # 之前缺失的
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
np.set_printoptions(threshold=np.inf)  # 设置numpy打印阈值为无限大
from Dataset1 import MultiFileBatchDataset
from Combined import Combined_Dataset3,zero_4d_based_on_a2,custom_collate_fn,Combined_Dataset2
from Net import MyModel11,MyModel9,MyModel13,MyModel14,MyModel18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.chdir('/mnt/hdb/LXH/data/')


directory_path = 'T850'

# 获取目录下所有符合条件的文件路径
input_paths = [os.path.join(directory_path, filename)
               for filename in os.listdir(directory_path)
               if filename.endswith('.nc')]
input_paths.sort()
input_paths = input_paths[4:76]
print(input_paths)
input_variable_name = 'T850'  # 变量名
input_batch_size = 64  # 批次大小

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qie = False
input_dataset1 = MultiFileBatchDataset(input_paths, input_variable_name, input_batch_size, device=device,
                                       apply_mean=False, mean_dims=(1, 2),replace_nan_with_zero=True,
                                       apply_interpolation=True, target_size=(72, 144),qie=qie,
                                       resample_daily=False,var_name='T850',normalize_skt = True,
                                       time_last= True,skt_std = False,
                                       file_idx = 0,max = 0,num = [])
# 创建自定义 Dataset
#dataset1 = MultiFileBatchDataset(file_paths, variable_name, batch_size, device=device)

directory_path = 'TSLO'

# 获取目录下所有符合条件的文件路径
input_paths2 = [os.path.join(directory_path, filename)
               for filename in os.listdir(directory_path)
               if filename.endswith('.nc.nc')]
input_paths2.sort()
input_paths2 = input_paths2[4:76]
print(input_paths2)
input_variable_name2 = 'TSOI_10CM'  # 变量名
input_batch_size2 = 64  # 批次大小

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qie = False
input_dataset2 = MultiFileBatchDataset(input_paths2, input_variable_name2, input_batch_size2, device=device,
                                       replace_nan_with_zero = False,
                                       apply_mean=False, mean_dims=(1, 2),apply_interpolation=False,
                                       target_size=(72, 144),qie=qie,replace_nan_with_zero_0 = True,
                                       resample_daily=False,var_name='TSOI_10CM',
                                       normalize_slt = True,slt_std = False,
                                       time_last = True,file_idx = 0,max = 0,num = [])

directory_path = 'TAL'

# 获取目录下所有符合条件的文件路径
output_paths = [os.path.join(directory_path, filename)
               for filename in os.listdir(directory_path)
               if filename.endswith('.nc.nc')]
output_paths.sort()
output_paths = output_paths[4:76]
print(output_paths)
output_variable_name = 'TLAI'  # 变量名
output_batch_size = 64  # 批次大小

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qie = False
output_dataset = MultiFileBatchDataset(output_paths, output_variable_name, output_batch_size, device=device,
                                       apply_mean=True, mean_dims=(1, 2),replace_nan_with_zero=True,
                                       normalize_lai=True,
                                       lai_std=True,time_last= True,replace_nan_with_zero_0 = True,
                                       qie=qie,file_idx = 0,max = 0,num = [])

train_combined_dataset = Combined_Dataset3(input_dataset1,input_dataset2,output_dataset)

train_loader = DataLoader(train_combined_dataset, batch_size=1)

# for normalized_diff, targets in train_loader:
#     print(normalized_diff)
#     print(targets)

xdim, ydim, cdim = 72, 144, 2  # 输入维度
max_batches_per_epoch = 9030  # 每个 epoch 中的最大训练批次数

import csv
import multiprocessing
# torch.set_printoptions(threshold=float('inf'))

op_train = 'on'  # 模拟训练标志
if op_train == 'on':
    # 初始化模型
    model = MyModel14(xdim, ydim, cdim)
    # model.half()
    model = model.to(device)
    # 初始化优化器
    initial_lr = 0.0001
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)


    # -------------------------------------------------------------------------------------
    # # #
    checkpoint_path = '/home/gis/lxh/Model/Adam22/Model_epoch_9.pth'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch']
        # 如果保存了学习率，将其恢复
    current_lr = 0.0001 # 从 checkpoint 中读取学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    # ------------------------------------------------------------------------------------------

    num_epochs = 200  # 减少训练轮数以加快示例速度
    history = {'loss': []}
    batch_interval = 30  # 每 5 个批次输出一次平均损失
    all_batch_losses = []  # 用于存储每个 batch_interval 后的损失

    csv_file_path = '/home/gis/lxh/Model/Adam22/losses.csv'
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['epoch', 'batch', 'loss'])  # 写入表头
    model.train()

    for epoch in range(start_epoch, num_epochs):

        epoch_loss = 0.0  # 初始化每个 epoch 的损失
        batch_count = 0  # 计数器，用于追踪每多少个批次输出一次损失
        per_epoch = 0
        train_combined_dataset.reset_indices()
        for inputs1,inputs2, targets in train_loader:
            # print('1')
            torch.autograd.set_detect_anomaly(True)
            inputs1,inputs2, targets = inputs1.to(device),inputs2.to(device), targets.to(device)

            inputs = torch.stack([inputs1, inputs2], dim=2)

            inputs = inputs.view(-1, 2, 72, 144)  # 或使用 reshape

            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print("Warning: inputs contain NaN or Inf")
            targets = targets.view(-1, 1)  # 或使用 reshape
            # print(targets[0:20, :])
            # print(inputs[0, 1, 24:50, :])

            optimizer.zero_grad()
            inputs = inputs.float()  # 转成float32
            targets = targets.float()

            # print(inputs[200, 0, 40:50, :])
            # print(inputs[200, 1, 40:50, :])
            print(inputs[:, 0, :, :].mean())
            print(inputs[:, 1, :, :].mean())

            outputs = model(inputs)
            print("原始输出形状:", outputs[30:40,:])

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Warning: outputs contain NaN or Inf")
                print(outputs)
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                print("Warning: targets contain NaN or Inf")
                print(targets)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            per_epoch += 1

            current_lr = optimizer.param_groups[0]['lr']

            # 每经过 batch_interval 个批次输出一次平均损失
            if batch_count % batch_interval == 0:
                avg_batch_loss = epoch_loss / batch_count  # 当前批次的平均损失
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{per_epoch}/{len(train_loader)}], Average Loss: {avg_batch_loss:.8f}')
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{per_epoch}/{len(train_loader)}], Learning Rate: {current_lr:.8e}')

                # 保存每个 batch_interval 后的损失到 CSV 文件
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch + 1, per_epoch, avg_batch_loss])  # 保存 epoch, batch, loss

                # 重置 epoch_loss 和 batch_count
                # epoch_loss = 0
                # batch_count = 0
                print(f'Model is running on: {next(model.parameters()).device}')
                if torch.cuda.is_available():
                    print(f'GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9} GB')
                    print(f'GPU Memory Cached: {torch.cuda.memory_reserved(device) / 1e9} GB')

            if per_epoch >= max_batches_per_epoch:
                print(f'Epoch [{epoch + 1}/{num_epochs}] ended early at batch {batch_count}/{len(train_loader)}')
                break

        avg_loss = epoch_loss / max_batches_per_epoch  # 当前 epoch 的平均损失
        history['loss'].append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.8f}')

        csv_file = '/home/gis/lxh/Model/Adam22/Epoch_losses.csv'
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow(['epoch', 'train_loss'])
            # 写入当前 epoch 和对应的损失
            writer.writerow([epoch + 1, avg_loss])

        # 调整学习率
        scheduler.step(avg_loss)
        # 保存当前学习率
        current_lr = optimizer.param_groups[0]['lr']


        # 每 1 个 epoch 保存一次模型
        model_save_path = f'/home/gis/lxh/Model/Adam22/Model_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch + 1,  # 保存当前 epoch
            'model_state_dict': model.state_dict(),  # 保存模型的权重
            'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器的状态
            'history': history,  # 保存损失历史
            'lr': current_lr  # 保存当前学习率
        }, model_save_path)
        print(f'Model for epoch {epoch + 1} saved at {model_save_path}')

        # if torch.cuda.is_available():0
        #    print(f'GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9} GB')
        #    print(f'GPU Memory Cached: {torch.cuda.memory_reserved(device) / 1e9} GB')
