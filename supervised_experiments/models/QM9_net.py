# adapted from https://github.com/AvivNavon/nash-mtl/blob/main/experiments/quantum_chemistry/models.py

import torch
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.nn import NNConv, Set2Set


class QM9_enc(torch.nn.Module):
    def __init__(self, num_features=11, dim=64):
        super().__init__()
        self.dim = dim
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr="mean")
        self.gru = GRU(dim, dim)
        self.gru.flatten_parameters()
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)

    def forward(self, data, mask=None):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        features = F.relu(self.lin1(out))

        return features, None


class QM9_dec(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        self.fc = torch.nn.Linear(self.dim, 1)

    def forward(self, x, mask=None):
        return self.fc(x), None
