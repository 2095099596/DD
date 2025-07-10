
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.nn.init as init

# class ConvSet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvSet, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
#         nn.init.xavier_uniform_(self.conv.weight)  # Xavier 初始化
#         self.activation = nn.Tanh()
#
#     def forward(self, x):
#         x = self.conv(x)
#         return self.activation(x)

class ConvSet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='tanh', reg=None):
        super(ConvSet, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.activation = self.get_activation(activation)
        self.reg = reg
        # 权重初始化：使用Xavier均匀初始化
        init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)
    def get_activation(self, act):
        if act == 'tanh':
            return torch.tanh  # 推荐使用 torch.tanh 替代 F.tanh（已弃用）
        elif act == 'sigmoid':
            return torch.sigmoid  # 同理，使用 torch.sigmoid
        else:
            return lambda x: x  # Identity
    def forward(self, x):
        x = self.conv(x)
        if self.reg:
            # 这里可以实现正则化逻辑
            pass
        return self.activation(x)


# class DenseSet(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(DenseSet, self).__init__()
#         self.dense = nn.Linear(in_features, out_features)
#         nn.init.xavier_uniform_(self.dense.weight)  # Xavier 初始化
#         self.activation = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.dense(x)
#         return x

class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        return self.pool(x)


class DenseSet(nn.Module):
    def __init__(self, in_features, out_features, activation='sigmoid', reg=None):
        super(DenseSet, self).__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.activation = self.get_activation(activation)
        self.reg = reg
    def get_activation(self, act):
        if act == 'sigmoid':
            return F.sigmoid
        elif act == 'tanh':
            return F.tanh
        else:
            return None
    def forward(self, x):
        x = self.dense(x)
        if self.activation is not None:  # Only call if activation is not None
            x = self.activation(x)
        if self.reg:  # Regularization logic can be implemented if necessary
            pass
        return x

class MyModel13(nn.Module):
    def __init__(self, xdim, ydim, cdim):
        super(MyModel13, self).__init__()
        self.conv1 = ConvSet(cdim, 32, kernel_size=3, activation='tanh')
        self.pool1 = MaxPool()
        self.conv2 = ConvSet(32, 32, kernel_size=3, activation='tanh')
        self.pool2 = MaxPool()
        self.conv3 = ConvSet(32, 16, kernel_size=3, activation='tanh')
        self.conv4 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.conv5 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(16 * (xdim // 4) * (ydim // 4), 32, activation='sigmoid')
        self.output = DenseSet(32, 1, activation=None)
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x



class MyModel9(nn.Module):
    def __init__(self, xdim, ydim, cdim):
        super(MyModel9, self).__init__()
        self.conv1 = ConvSet(cdim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)

        self.conv2 = ConvSet(32, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(16 * 6 * 8, 32)
        self.output = DenseSet(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.output(x)
        return x


class MyModel10(nn.Module):
    def __init__(self, xdim, ydim, cdim):
        super(MyModel10, self).__init__()
        self.conv1 = ConvSet(cdim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)

        self.conv2 = ConvSet(32, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(32 * 6 * 8, 32)
        self.output = DenseSet(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.output(x)
        return x

class MyModel11(nn.Module):
    def __init__(self, xdim, ydim, cdim):
        super(MyModel11, self).__init__()
        self.conv1 = ConvSet(cdim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)

        self.conv2 = ConvSet(32, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=1)

        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(32 * 13 * 16, 32)
        self.output = DenseSet(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.output(x)
        return x


class MaxPool14(nn.Module):
    def __init__(self):
        super(MaxPool14, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.pool(x)

class MyModel14(nn.Module):
    def __init__(self, xdim, ydim, cdim):
        super(MyModel14, self).__init__()
        self.conv1 = ConvSet(cdim, 64, kernel_size=3, activation='tanh')
        self.pool1 = MaxPool14()
        self.conv2 = ConvSet(64, 32, kernel_size=3, activation='tanh')
        self.pool2 = MaxPool14()
        self.conv3 = ConvSet(32, 16, kernel_size=3, activation='tanh')
        self.conv4 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.conv5 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(16 * (xdim // 1) * (ydim // 1), 32, activation='sigmoid')
        self.output = DenseSet(32, 1, activation=None)
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x


class MaxPool18(nn.Module):
    def __init__(self):
        super(MaxPool14, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        return self.pool(x)

class MyModel14(nn.Module):
    def __init__(self, xdim, ydim, cdim):
        super(MyModel14, self).__init__()
        self.conv1 = ConvSet(cdim, 64, kernel_size=3, activation='tanh')
        self.pool1 = MaxPool14()
        self.conv2 = ConvSet(64, 32, kernel_size=3, activation='tanh')
        self.pool2 = MaxPool14()
        self.conv3 = ConvSet(32, 16, kernel_size=3, activation='tanh')
        self.conv4 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.conv5 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(16 * (xdim // 1) * (ydim // 1), 32, activation='sigmoid')
        self.output = DenseSet(32, 1, activation=None)
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x


class MyModel15(nn.Module):
    def __init__(self, xdim, ydim, cdim):
        super(MyModel15, self).__init__()
        self.conv1 = ConvSet(cdim, 64, kernel_size=3, activation='tanh')
        self.pool1 = MaxPool14()
        self.conv2 = ConvSet(64, 32, kernel_size=3, activation='tanh')
        self.pool2 = MaxPool14()
        self.conv3 = ConvSet(32, 16, kernel_size=3, activation='tanh')
        self.conv4 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.conv5 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(16 * (xdim // 1) * (ydim // 1), 32, activation='sigmoid')
        self.output = DenseSet(32, 1, activation=None)
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x



class MyModel18(nn.Module):
    def __init__(self, xdim, ydim, cdim):
        super(MyModel18, self).__init__()
        self.conv1 = ConvSet(cdim, 128, kernel_size=3, activation='tanh')
        self.pool1 = MaxPool14()
        self.conv2 = ConvSet(128, 64, kernel_size=3, activation='tanh')
        self.pool2 = MaxPool14()
        self.conv3 = ConvSet(64, 32, kernel_size=3, activation='tanh')
        self.conv4 = ConvSet(32, 32, kernel_size=3, activation='tanh')
        self.conv5 = ConvSet(32, 16, kernel_size=3, activation='tanh')
        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(16 * (xdim // 1) * (ydim // 1), 32, activation='sigmoid')
        self.output = DenseSet(32, 1, activation=None)
    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv5(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.dense1(x)
        # print(x.shape)
        x = self.output(x)
        # print(x.shape)
        return x

class LSTMModel1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    def forward(self, x):
        # print(x.shape)
        out, (h_n, c_n) = self.lstm(x)
        return out

class LSTMModel2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel2, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 12, num_layers, batch_first=True)
        self.lstm2 = LSTMModel1(12, hidden_size, num_layers)
    def forward(self, x):

        x, (h_n, c_n)= self.lstm1(x)

        out = self.lstm2(x)
        return out


class CnnLSTMModel(nn.Module):
    def __init__(self,xdim,ydim,cdim,input_size1,input_size2,hidden_size):
        super(CnnLSTMModel, self).__init__()
        self.lstm1 = LSTMModel1(input_size1, hidden_size,1)
        self.lstm2 = LSTMModel2(input_size2, hidden_size,1)
        self.conv1 = ConvSet(cdim, 32, kernel_size=3, activation='tanh')
        self.pool1 = MaxPool14()
        self.conv2 = ConvSet(32, 32, kernel_size=3, activation='tanh')
        self.pool2 = MaxPool14()
        self.conv3 = ConvSet(32, 16, kernel_size=3, activation='tanh')
        self.conv4 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.conv5 = ConvSet(16, 16, kernel_size=3, activation='tanh')
        self.flatten = nn.Flatten()
        self.dense1 = DenseSet(16 * 96 * 144, 32, activation='sigmoid')
        self.output = DenseSet(32, 1, activation=None)

    def forward(self, x1,x2,x3):
        x1 = x1.view(2, 12, -1)   # Shape: (2, 12, 13824)
        x1 = x1.permute(0, 2, 1)  # Shape: (2, 13824, 12)
        x1 = self.lstm1(x1)       # Shape: (2, 13824, 1)
        x1 = x1.view(2, 1, 96, 144, 1) # Shape: 2, 1, 96, 144, 1)
        x2 = x2.view(2, 365, -1)  # Shape: (2, 365, 13824)
        x2 = x2.permute(0, 2, 1)  # Shape: (2, 13824, 365)
        x2 = self.lstm2(x2)       # Shape: (2, 13824, 1)
        x2 = x2.view(2, 1, 96, 144, 1) # Shape: 2, 1, 96, 144, 1)
        x = torch.cat((x1,x2,x3),1)
        x = x.squeeze(-1)  # -1 表示最后一维
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.output(x)
        return x