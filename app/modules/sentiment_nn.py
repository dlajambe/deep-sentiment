import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentNet(nn.Module):
    def __init__(self, n_embed: int, block_size: int):
        super(SentimentNet, self).__init__()
        self.linear = nn.Linear(n_embed * block_size, 128)
        self.output = nn.Linear(128, 1)
        self.block_size = block_size
        self.n_embed = n_embed

    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear(x))
        y_p = F.sigmoid(self.output(x))
        return y_p
