from typing import List

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        label_list: List[int],
        input_channel: int = 2,
        device="cpu"
    ):
        super().__init__()
        self.label_list = label_list
        self.input_channel = input_channel
        self.device = device
        self.fc = nn.Linear(self.input_channel, len(self.label_list))
    
    def forward(self, z):
        return self.fc(z)

    def loss(self, z, label):
        t = self.forward(z)
        criterion = nn.CrossEntropyLoss()
        return criterion(t, label)

    def accuracy(self, x: torch.Tensor, label: torch.Tensor):
        t = self.forward(x)
        acc = 0
        data_length = len(t)
        for i in range(data_length):
            if self.label_list[torch.argmax(t[i]).item()] == label[i]:
                acc += 1
        return acc / data_length * 100