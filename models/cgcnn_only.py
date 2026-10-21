# 仅使用晶体结构图，不使用描述符。

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import CGConv, global_mean_pool


class CGCNNOnly(nn.Module):

    def __init__(self, node_dim=5):
        super().__init__()

        self.lin0 = nn.Linear(node_dim, 64)

        self.conv1 = CGConv(64, dim=1)
        self.conv2 = CGConv(64, dim=1)

        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):

        x = F.relu(self.lin0(data.x))

        x = self.conv1(x, data.edge_index, data.edge_attr)
        x = self.conv2(x, data.edge_index, data.edge_attr)

        g = global_mean_pool(x, data.batch)

        out = self.head(g)

        return g, out