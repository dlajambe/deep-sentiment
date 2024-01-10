import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import Adam
from modules.data_preprocessing import TokenDataset

class SentimentNet(nn.Module):
    def __init__(self, n_embed: int, block_size: int):
        super(SentimentNet, self).__init__()
        self.linear1 = nn.Linear(n_embed * block_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)
        self.block_size = block_size
        self.n_embed = n_embed
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        y_p = F.sigmoid(self.output(x))
        return y_p
    
def train_model(
        model: SentimentNet,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        max_epochs: int,
        batch_size: int,
        lr: float):
    """Trains a SentimentNet neural network using the provided
    hyperparameters.

    The x data must be provided in the shape expected by the
    SentimentNet model, i.e. (N, T, C), where:
    - N: Number of token sequences in the data partition
    - T: Bock size, i.e. the number of tokens per sequence
    - C: Number of embeddings per token

    The y data must be of shape (N, 1).

    Parameters
    ----------
    model : SentimentNet
        The neural network to be trained.

    x_train : Tensor
        The input data for the training partition.

    y_train : Tensor
        The target data for the training partition.

    x_val : Tensor
        The input data for the validation partition.

    y_val : Tensor
        The target data for the validation partition.

    max_epochs : int
        The maximum number of passes through the entire training dataset
        to be used for backpropagation.

    batch_size : int
        The number of samples to use for each iteration of
        backpropagation.

    lr : float
        The learning rate to be applied during training.
    """
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Creating a DataLoader makes it easy to iterate over batches during
    # training
    data_loader_train = DataLoader(
        TokenDataset(x_train, y_train),
        shuffle=True,
        drop_last=True,
        batch_size=batch_size)
    
    for i in range(max_epochs):
        for xb, yb in data_loader_train:
            optimizer.zero_grad()
            xb = xb.view(batch_size, -1)
            yb_p = model.forward(xb)
            loss = criterion.forward(yb_p, yb)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_train_p = model.forward(x_train.view(len(x_train), -1))
            loss_train = criterion.forward(y_train_p, y_train)
            
            y_val_p = model.forward(x_val.view(len(x_val), -1))
            loss_val = criterion.forward(y_val_p, y_val)

        print('\tEpoch: {0}\tLoss (train): {1}\tLoss (val): {2}'.format(
            i, loss_train.item(), loss_val.item()))