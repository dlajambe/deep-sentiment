import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import Adam
from modules.data_preprocessing import TokenDataset

class TransformerBlock(nn.Module):
    def __init__(self, n_embed: int, block_size: int, n_heads: int,
                 ff_proj_factor: int=4) -> None:
        super(TransformerBlock, self).__init__()
        if n_embed % n_heads != 0:
            raise ValueError('n_embed must be a multiple of n_heads')
        self.attention = nn.MultiheadAttention(n_embed, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embed, n_embed*ff_proj_factor),
            nn.ReLU(),
            nn.Linear(n_embed*ff_proj_factor, n_embed)
        )

        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention, _ = self.attention(x, x, x)
        attention = self.layer_norm_1(self.dropout(attention))
        features = self.layer_norm_2(
            self.dropout(self.feed_forward(attention)))
        return features

class SentimentNet(nn.Module):
    """A transformer-based language model for classifying the sentiment
    of English language text blocks."""
    def __init__(self, n_embed: int, block_size: int):
        super(SentimentNet, self).__init__()
        
        # Positional embeddings are required so that the location of
        # tokens within a block can be used to generate features
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.positions = torch.arange(block_size, device='cpu')

        self.trans = TransformerBlock(n_embed, block_size, 5)
        self.fc = nn.Linear(n_embed, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x + self.pos_embed(self.positions)
        features = self.trans(input)
        features = features.mean(dim=1)
        y_p = F.sigmoid(self.fc(features).view(x.shape[0],))
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
            yb_p = model.forward(xb)
            loss = criterion.forward(yb_p, yb)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_train_p = model.forward(x_train)
            loss_train = criterion.forward(y_train_p, y_train)
            
            y_val_p = model.forward(x_val)
            loss_val = criterion.forward(y_val_p, y_val)

        print('\tEpoch: {0}\tLoss (train): {1}\tLoss (val): {2}'.format(
            i, loss_train.item(), loss_val.item()))