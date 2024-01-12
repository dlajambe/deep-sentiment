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
    def __init__(self, n_embed: int, block_size: int, device: str):
        super(SentimentNet, self).__init__()
        if device != 'cuda' and device != 'cpu':
            raise ValueError(
                'Invalid device received. Must be "cuda" or "cpu"')
        # Positional embeddings are required so that the location of
        # tokens within a block can be used to generate features
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.positions = torch.arange(block_size, device=device)

        self.trans = TransformerBlock(n_embed, block_size, 5)
        self.fc = nn.Linear(n_embed, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x + self.pos_embed(self.positions)
        features = self.trans(input)

        # A sentiment logit has been calculated for each token in a
        # sequence, but we require only one class logit per sequence
        # Using the average of all logits to calculate the class logit
        # of each sequence. 
        features = features.mean(dim=1)

        # The logits must be converted to probabilities with the sigmoid
        # function because the BCE loss function requires them in this
        # form
        y_p = F.sigmoid(self.fc(features).view(x.shape[0],))
        return y_p

def calc_metrics(
        model: nn.Module,
        data_loader: DataLoader,
        device: str) -> tuple[float, float]:
    """Calculates the mean binary cross entropy loss and mean accuracy
    of a model over a dataset.

    Parameters
    ----------
    model : Module
        The PyTorch model whose loss and accuracy are to be calculated.

    data_loader : DataLoader
        The data loader containing the dataset over which the mean loss
        and accuracy are to be calculated.

    device : str {'cuda', 'cpu'}
        The device on which the calculations are to be performed.

    Returns
    -------
    loss_mean : float
        The mean loss of the model over the provided dataset.

    accuracy_mean : float
        The mean accuracy of the model over the provided dataset.
    """
    criterion = nn.BCELoss()

    # No need to waste time calculating gradients when calculating
    # metrics since backpropagation will not occur
    with torch.no_grad():
        losses = []
        accuracies = []
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            yb_p = model.forward(xb)
            labels = torch.round(yb_p)
            loss = criterion.forward(yb_p, yb)
            losses.append(loss.item())
            accuracies.append(torch.sum(labels == yb).item() / float(yb.shape[0]))
        loss_mean = sum(losses)/float(len(losses))
        accuracy_mean = sum(accuracies)/float(len(accuracies))
        return loss_mean, accuracy_mean

def train_model(
        model: SentimentNet,
        dataset_train: TokenDataset,
        dataset_val: TokenDataset,
        max_epochs: int,
        batch_size: int,
        lr: float,
        device: str) -> None:
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

    dataset_train : TokenDataset
        The training dataset.

    dataset_val : TokenDataset
        The validation dataset.

    max_epochs : int
        The maximum number of passes through the entire training dataset
        to be used for backpropagation.

    batch_size : int
        The number of samples to use for each iteration of
        backpropagation.

    lr : float
        The learning rate to be applied during training.

    device : str ['cuda', 'cpu']
        The device on which to train the model.
    """
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Creating a DataLoader makes it easy to iterate over batches during
    # training and performance calculation
    data_loader_train = DataLoader(
        dataset_train,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size)
    
    data_loader_val = DataLoader(
        dataset_val,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size)
    
    loss_val_min = float('inf')
    best_model = {}
    for i in range(max_epochs):
        for xb, yb in data_loader_train:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            yb_p = model.forward(xb)
            loss = criterion.forward(yb_p, yb)
            loss.backward()
            optimizer.step()

        # Performance metrics are calculated at the end of each epoch
        model.eval()
        loss_train, accuracy_train = calc_metrics(
            model, data_loader_train, device)
        loss_val, accuracy_val = calc_metrics(
            model, data_loader_val, device)
        model.train()
        
        # To avoid overfitting, the model parameters that maximize
        # performance on the validation dataset are saved
        if loss_val < loss_val_min:
            loss_val_min = loss_val
            best_model = model.state_dict()

        print('\tEpoch: {0}\tLoss (train): {1}\tLoss (val): {2}\tAccuracy (train): {3}\tAccuracy (val): {4}'.format(
            i, round(loss_train, 4), round(loss_val, 4), 
            round(accuracy_train, 4), round(accuracy_val, 4)))
    
    # Training is now finished, so the model parameters are set to the
    # performance-maximizing set prior to function termination
    model.load_state_dict(best_model)