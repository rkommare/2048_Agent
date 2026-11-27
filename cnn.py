import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self,channels,conv_layers=3):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1,channels,3,padding='same'),nn.ReLU())
        while len(self.layers) < conv_layers*3:
            self.layers.append(nn.Conv2d(channels,channels,3,padding='same'))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(channels*16,channels*4))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(channels*4,channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(channels,4))

    def forward(self,x):
        return self.layers(x)