# Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py

import os
import os.path
import numpy as np
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.utils import remove_self_loops

"""
The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
about 130,000 molecules with 19 regression targets.
Each molecule includes complete spatial information for the single low
energy conformation of the atoms in the molecule.
In addition, we provide the atom features from the `"Neural Message
Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| Target | Property                         | Description                                                                       | Unit                                        |
+========+==================================+===================================================================================+=============================================+
| 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
"""


task_to_qm9_idx = {
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "epsilon": 4,
    "r2": 5,
    "zpve": 6,
    "U0": 7,
    "U": 8,
    "H": 9,
    "G": 10,
    "Cv": 11,
    "U0Atom": 12,
    "UAtom": 13,
    "HAtom": 14,
    "GAtom": 15,
    "A": 16,
    "B": 17,
    "C": 18,
}


class MyTransform(object):
    def __init__(self, target: list):
        self.target = target

    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, self.target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class QM9_dataset(data.Dataset):
    # """https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html"""
    def __init__(self, root, tasks, split="train"):
        self.targets = [task_to_qm9_idx[t] for t in tasks]

        self.transform = T.Compose([MyTransform(self.targets), Complete(), T.Distance(norm=False)])
        self.dataset = QM9(root, transform=self.transform)

        # Normalize dataset
        mean = self.dataset.data.y[:, self.targets].mean(dim=0, keepdim=True)
        self.std = self.dataset.data.y[:, self.targets].std(dim=0, keepdim=True)
        self.dataset.data.y[:, self.targets] = (self.dataset.data.y[:, self.targets] - mean) / self.std

        if os.path.exists(os.path.join(root, "indices.txt")):
            indices = np.loadtxt(os.path.join(root, "indices.txt"), dtype=int)
        else:
            indices = list(range(len(self.dataset)))
            random.Random(0).shuffle(indices)
            np.savetxt(os.path.join(root, "indices.txt"), indices, fmt="%d")

        test_indices = indices[:10000]
        val_indices = indices[10000:20000]
        train_indices = indices[20000:]

        train_dataset = data.Subset(self.dataset, train_indices)
        val_dataset = data.Subset(self.dataset, val_indices)
        test_dataset = data.Subset(self.dataset, test_indices)

        if split == "train":
            self.selected_dataset = train_dataset
        elif split == "val":
            self.selected_dataset = val_dataset
        elif split == "test":
            self.selected_dataset = test_dataset
        else:
            raise NotImplementedError(f"{split} not supported for QM9 dataset")

    def get_std(self, idx):
        return self.std[0, idx]


# if __name__ == "__main__":
# qm9_target_dict = {
#     0: "mu",
#     1: "alpha",
#     2: "homo",
#     3: "lumo",
#     5: "r2",
#     6: "zpve",
#     7: "U0",
#     8: "U",
#     9: "H",
#     10: "G",
#     11: "Cv",
# }

# tasks = ["mu", "alpha", "homo", "lumo", "r2", "zpve", "U0", "U", "H", "G", "Cv"]
# tasks = ["mu"]
# dataset = QM9_dataset("datasets/QM9/", tasks)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# data_loader = DataLoader(dataset.selected_dataset, batch_size=120, shuffle=True, num_workers=2)

# dim = 64
# model = Net(n_tasks=len(tasks), num_features=11, dim=dim).to(DEVICE)
# optimizer = torch.optim.Adam(model.parameters())

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

# for epoch in tqdm(range(100)):
#     mean_loss = 0
#     print("len dataloader: ", len(data_loader))
#     for j, batch in tqdm(enumerate(data_loader)):
#         model.train()
#         batch = batch.to(DEVICE)

#         optimizer.zero_grad()

#         out, features = model(batch, return_representation=True)

#         losses = F.mse_loss(out, batch.y, reduction="none").mean()
#         losses.backward()

#         optimizer.step()
#         scheduler.step()
#         mean_loss += losses.item()

#     print("mean_loss: ", mean_loss)
