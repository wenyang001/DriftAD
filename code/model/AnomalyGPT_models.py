import torch
from torch import nn
import numpy as np
# from datas.dataset_3d import  *
from torch.nn import functional as F


class Normalize(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)

    
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        self.relu = nn.ReLU()

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i].transpose(0, 1)
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
            tokens[i] = self.relu(tokens[i])
        return tokens



## FC

class Adapter(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Adapter, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        return x


## kernel-aware
class MMCI(nn.Module):
    def __init__(self, input_channels=1024, output_channels=1024):
        super(MMCI, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 4))
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=(4, 1))
        self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=2)
        self.conv4 = nn.Conv2d(input_channels, output_channels, kernel_size=3)
        # self.activation = nn.ReLU()

    def forward(self, x):
        size = x[0].size()[2:]
        outputs = []
        for i in range(len(x)):
            out1 = self.conv1(x[i])
            out2 = self.conv2(x[i])
            out3 = self.conv3(x[i])
            out4 = self.conv4(x[i])
            out1 = F.interpolate(out1, size=size, mode='bilinear', align_corners=True)
            out2 = F.interpolate(out2, size=size, mode='bilinear', align_corners=True)
            out3 = F.interpolate(out3, size=size, mode='bilinear', align_corners=True)
            out4 = F.interpolate(out4, size=size, mode='bilinear', align_corners=True)
            feature = (out1 + out2 + out3 + out4)/4.
            # feature = self.activation(feature)
            outputs.append(feature)

        return outputs

