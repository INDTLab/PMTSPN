import torch
import torch.nn as nn

class pmtspn(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x1, x2):
        return torch.sum(x1-x2)