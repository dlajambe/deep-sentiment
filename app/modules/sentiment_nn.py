import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

class TransformerBlock(nn.Module):
    """Implements a single transformer block, as described in the 
    "Attention Is All You Need" paper.
    """
    def __init__(self, n_embed: int, block_size: int, n_heads: int,
                 dropout_frac: float, ff_proj_factor: int=4) -> None:
        super(TransformerBlock, self).__init__()
        if n_embed % n_heads != 0:
            raise ValueError('n_embed must be a multiple of n_heads')
        self.attention = nn.MultiheadAttention(
            n_embed, n_heads,
            dropout=dropout_frac,
            batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embed, n_embed*ff_proj_factor),
            nn.ReLU(),
            nn.Linear(n_embed*ff_proj_factor, n_embed),
            nn.Dropout(dropout_frac)

        )

        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention, _ = self.attention(x, x, x)
        attention = self.layer_norm_1(attention)
        features = self.layer_norm_2(self.feed_forward(attention))
        return features

class SentimentNet(nn.Module):
    """A language model used to classify the sentiment of English
    language movie reviews as positive or negative.

    The model uses a transformer-based architecture. An initial
    transformer block extracts features from the input sequence, which
    are then converted to a class logit by a sequence of fully connected
    linear layers.
    
    Attributes
    ----------
    pos_embed : Embeddings
        Transforms a token's position in a sequence into a vectorized
        representation.

    positions : Tensor
        A generic tensor used to create a position token for each token
        in an input sequence.

    dropout : Dropout
        The dropout module used to regularize the training process by
        randomly deactivating nodes during model prediction.

    trans : TransformerBlock
        A transformer block used to extract features from an input token
        sequence.

    fc : Sequential
        The fully connected linear layers used to transform the features
        generated by the transformer block into class logits.
    """
    def __init__(
            self, 
            n_embed: int, 
            block_size: int, 
            n_heads: int,
            dropout_frac: float, 
            device: str) -> None:
        super(SentimentNet, self).__init__()
        if device != 'cuda' and device != 'cpu':
            raise ValueError(
                'Invalid device received. Must be "cuda" or "cpu"')
        # Positional embeddings are required so that the location of
        # tokens within a block can be used to generate features
        self.pos_embed = nn.Embedding(block_size, n_embed)
        self.positions = torch.arange(block_size, device=device)
        self.dropout = nn.Dropout(dropout_frac)

        self.trans = TransformerBlock(
            n_embed, block_size, n_heads, dropout_frac)

        # The fully connected layers are used to convert features
        # generated by the transformer block(s) into a sentiment
        # prediction
        self.fc = nn.Sequential(
            nn.Linear(n_embed, int(n_embed/2)),
            nn.ReLU(),
            self.dropout,
            nn.Linear(int(n_embed/2), int(n_embed/4)),
            nn.ReLU(),
            self.dropout,
            nn.Linear(int(n_embed/4), 1)
        )

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
    losses = []
    accuracies = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            yb_p = model.forward(xb)
            loss = criterion.forward(yb_p, yb)
            losses.append(loss.item())
            labels = torch.round(yb_p)
            accuracies.append(torch.sum(labels == yb).item() / float(yb.shape[0]))
    loss_mean = sum(losses)/float(len(losses))
    accuracy_mean = sum(accuracies)/float(len(accuracies))
    return loss_mean, accuracy_mean

def train_model(
        model: nn.Module,
        dataset_train: Dataset,
        dataset_val: Dataset,
        max_epochs: int,
        batch_size: int,
        lr: float,
        device: str) -> None:
    """Trains a neural network using the provided hyperparameters.

    The x data must be provided in the shape expected by the
    SentimentNet model, i.e. (N, T, C), where:
    - N: Number of token sequences in the data partition
    - T: Bock size, i.e. the number of tokens per sequence
    - C: Number of embeddings per token

    The y data must be of shape (N, 1).

    Parameters
    ----------
    model : Module
        The neural network to be trained.

    dataset_train : Dataset
        The training dataset.

    dataset_val : Dataset
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
        new_best = False
        if loss_val < loss_val_min:
            loss_val_min = loss_val
            best_model = model.state_dict()
            new_best = True
        output_msg = (
            '\tEpoch: {}\tLoss (train): {:.4f}\tLoss (val): {:.4f}\t'
            'Accuracy (train): {:.4f}\tAccuracy (val): {:.4f}'.
            format(
                i, 
                round(loss_train, 4), 
                round(loss_val, 4), 
                round(accuracy_train, 4),
                round(accuracy_val, 4)))
        if new_best:
            output_msg += '\t*'
        print(output_msg)
    
    # Training is now finished, so the model parameters are set to the
    # performance-maximizing set prior to function termination
    model.load_state_dict(best_model)
    model.eval()