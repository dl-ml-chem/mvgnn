# 只使用人工描述符。

import torch.nn as nn


class DescriptorOnly(nn.Module):

    def __init__(self, input_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, data):

        out = self.net(data.desc)

        return data.desc, out