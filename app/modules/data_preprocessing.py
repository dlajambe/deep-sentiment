from string import punctuation
import torch
from torch.utils.data import Dataset

from torchtext.vocab import GloVe

def tokenize_review(review_raw: str, block_size: int) -> list[str]:
    """Converts a review to a token by separating the review into a list
    of preprocessed words. 
    
    Preprocessing steps applied:
    - Artifact removal
    - Punctuation removal
    - Transformation to lower case
    - Splitting into words using spaces
    - Padding or truncating the review to a length of block_size

    Parameters
    ----------
    review_raw : str
        The review to be preprocessed.

    block_size : int
        The number of words to include in each tokenized review. Reviews
        are truncated or padded to a length of block_size.

    Returns
    -------
    review_new : list[str]
        A list of strings (words) representing the final, tokenized
        review.
    """

    # The IMDb dataset contains <br /> before eac sentence; these do
    # not add semantic value and can be removed
    review_new = review_raw.replace('<br />', ' ')

    # Punctuation is removed to standardize the words, although this may
    # result in a loss of information
    review_new = ''.join(
        [char for char in review_new if char not in punctuation])
    review_new = review_new.lower().split()[:block_size]

    # Reviews that are too short must be padded with empty strings so
    # that all reviews are the same length (block_size) during training
    if len(review_new) < block_size:
        review_new.extend([''] * (block_size - len(review_new)))
    if len(review_new) != block_size:
        raise ValueError('Preprocessed review is not of length block_size')
    return review_new

class TokenDataset(Dataset):
    """Stores a tensor-based dataset containing tokenized x and y data.

    - N: Total number of token sequences in the dataset
    - T: Bock size, i.e. the number of tokens per sequence
    - C: Number of embeddings per token

    Attributes
    ----------
    _reviews : list[list[str]]
        A list of lists, each contianing the string representation of a
        single sample. These are converted to torch tensors on-the-fly.

    _y : Tensor
        A tensor of shape (N,) containing the class labels for each
        sample in _x.
    """
    def __init__(
            self, 
            reviews: list[list[str]], 
            sentiment: list[str],
            n_embed: int) -> None:
        if len(reviews) != len(sentiment):
            raise ValueError(
                'reviews and sentiment must have the same number of samples')
        
        y = torch.zeros(size=(len(reviews),))
        for i in range(len(y)):
            y[i] = sentiment[i]
        self._reviews = reviews
        self._y = y

        # The movie reviews dataset is too small to train word emebddings,
        # so pretrained embeddings (GloVe) are used instead
        self.glove = GloVe(name="6B", dim=n_embed)

    def __len__(self) -> int:
        return self._y.shape[0]

    def __getitem__(self, idx: int) -> [torch.Tensor, torch.Tensor]:
        x = self.glove.get_vecs_by_tokens(self._reviews[idx])
        return x, self._y[idx]