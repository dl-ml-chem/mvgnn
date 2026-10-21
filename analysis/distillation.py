# 教师模型蒸馏到轻量学生模型。

import torch
import torch.nn as nn
import torch.optim as optim


class StudentNet(nn.Module):

    def __init__(self, input_dim=8):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def run_distillation(
    teacher,
    loader,
    device="cpu",
    epochs=100
):

    student = StudentNet().to(device)

    optimizer = optim.Adam(
        student.parameters(),
        lr=1e-3
    )

    loss_fn = nn.MSELoss()

    teacher.eval()

    for epoch in range(epochs):

        for batch in loader:

            batch = batch.to(device)

            with torch.no_grad():
                _, t_pred = teacher(batch)

            s_pred = student(batch.desc)

            loss = loss_fn(s_pred, t_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Student model trained.")

    return student