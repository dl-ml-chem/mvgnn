import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import CGConv, global_mean_pool


class MultiViewCGCNN(nn.Module):

    def __init__(self, node_dim=8, desc_dim=8, edge_dim=16):
        super().__init__()

        self.lin0 = nn.Linear(node_dim, 128)
        self.bn0 = nn.BatchNorm1d(128)

        self.conv1 = CGConv(128, dim=edge_dim)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.2)

        self.conv2 = CGConv(128, dim=edge_dim)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = CGConv(128, dim=edge_dim)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)

        self.desc_net = nn.Sequential(
            nn.Linear(desc_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, data):

        # Graph branch
        x = F.relu(self.bn0(self.lin0(data.x)))

        x = F.relu(self.bn1(self.conv1(x, data.edge_index, data.edge_attr)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x, data.edge_index, data.edge_attr)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.conv3(x, data.edge_index, data.edge_attr)))
        x = self.drop3(x)

        g = global_mean_pool(x, data.batch)

        # Descriptor branch
        d = self.desc_net(data.desc)

        # Fusion
        fusion = torch.cat([g, d], dim=1)

        out = self.head(fusion)

        return fusion, out
