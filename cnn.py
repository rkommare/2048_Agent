import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.temp = nn.Linear(16,4)

    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.temp(x)
        return x