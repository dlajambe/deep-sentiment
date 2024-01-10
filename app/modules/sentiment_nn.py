import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import Adam
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
    
def train_model(
        model: nn.Module, 
        data_loader: DataLoader,
        max_epochs: int,
        batch_size: int,
        lr: float):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for i in range(max_epochs):
        losses = []
        for xb, yb in data_loader:
            optimizer.zero_grad()
            xb = xb.view(batch_size, -1)
            yb_p = model.forward(xb)
            loss = criterion.forward(yb_p, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_mean = sum(losses)/float(len(losses))
        print('Epoch {0} loss: {1}'.format(i, loss_mean))