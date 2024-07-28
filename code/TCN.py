import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

#Sequence trimming
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

#TCN block
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#TCN model
class TemporalConvNet(nn.Module):
    def __init__(self, TCN_num_inputs, TCN_num_channels, TCN_kernel_size=3, TCN_dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(TCN_num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i #Dilation rate：1,2,4,8,...
            in_channels = TCN_num_inputs if i == 0 else TCN_num_channels[i - 1]
            out_channels = TCN_num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, TCN_kernel_size, stride=1, dilation=dilation_size,padding=(TCN_kernel_size - 1) * dilation_size, dropout=TCN_dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        out=self.network(inputs)#(B,c,l)
        return out