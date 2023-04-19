import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchaudio

class StupidModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.ReLU(),
            nn.Flatten(start_dim=1)
        )
        self.decoder = nn.Linear(8 * 16, num_classes)
    
    def forward(self, batch):
        return self.decoder(self.encoder(batch))
