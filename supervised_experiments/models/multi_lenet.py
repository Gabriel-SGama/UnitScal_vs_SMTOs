import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain


class MultiLeNetE(nn.Module):
    def __init__(self, p=0.0):
        super(MultiLeNetE, self).__init__()
        self._conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self._conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self._fc = nn.Linear(320, 50)
        # using PyTorch dropout instead of the one implemented by hand in https://github.com/isl-org/MultiObjectiveOptimization
        self.dropout = nn.Dropout(p=p)

        self.p = p

    def forward(self, x, mask=None):
        x = F.relu(F.max_pool2d(self._conv1(x), 2))
        x = self._conv2(x)
        x = self.dropout(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 320)
        x = F.relu(self._fc(x))
        return x, mask


class MultiLeNetC(nn.Module):
    def __init__(self, p=0.0):
        super(MultiLeNetC, self).__init__()
        self._fc1, self._fc2 = nn.Linear(50, 50), nn.Linear(50, 10)
        self.dropout = nn.Dropout(p=p)
        self.p = p

    def forward(self, x, mask=None):
        x = F.relu(self._fc1(x))
        x = self.dropout(x)
        x = self._fc2(x)
        return F.log_softmax(x, dim=1), mask


class MultiLeNetR(nn.Module):
    def __init__(self, p=0):
        super(MultiLeNetR, self).__init__()
        self._fc1, self._fc2 = nn.Linear(50, 256), nn.Linear(256, 784)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, mask=None):
        x = F.relu(self._fc1(x))
        x = self.dropout(x)
        x = self._fc2(x)
        x = x.view(-1, 28, 28)
        return x, mask
