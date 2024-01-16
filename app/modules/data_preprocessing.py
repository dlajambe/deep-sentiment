from string import punctuation
import torch
from torch.utils.data import Dataset

from torchtext.vocab import GloVe

def tokenize_review(review_raw: str, block_size: int) -> list[str]:
    """Converts a review to a token sequence by separating the review
    into a list of preprocessed words. 
    
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
    """Stores a text dataset containing tokenized x data and targets.
    The targets are stored as tensors, whereas the x-data are stored as
    strings and converted to tensors on-the-fly by GloVe word 
    embeddings. 
    
    Although it is slower to convert tokens to vector embeddings on-the-
    fly repeatedly, it is for medium to large datasets because
    converting the tokens to embedding vectors dramatically increases
    the dimensionality of the x-data. If the x tensors are calculated in
    advance, they must be stored in memrory, which can easily overload
    the computer's memory capacity.

    When converted to tensor format, the x-data is of shape (N, T, C),
    where:
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
    
class DataPreprocessor():
    """Tokenizes and vectorizes reviews into tensors so they can be fed
    as input into a neural network.

    If the review has a length of L after tokenization and n_embed = E,
    the final output tensor will be of shape (L, E).

    Attributes
    ----------
    n_embed : int {50, 100, 200, 300}
        The dimensionality of the vectorized space into which the
        tokenized sequence is to be projected by the embeddings.

    glove : GloVe
        The pretrained word embeddings used to vectorize the tokenized
        reviews.
    """
    def __init__(self, n_embed: int) -> None:
        self.glove = GloVe(name="6B", dim=n_embed)
        self.n_embed = n_embed

    def preprocess_review(self, review: str) -> torch.Tensor:
        """Fully preprocesses a review by tokenizing it and then
        transforming into vector embeddings.

        The end result is a 2-dimensional tensor that is ready to be fed
        as input into a neural network.

        Parameters
        ----------
        review : str
            The review to be preprocessed.

        Returns
        -------
        review_tensor : Tensor
            The tokenized and vectorized review.  
        """
        review_tokens = tokenize_review(review)
        review_tensor = self.glove.get_vecs_by_tokens(review_tokens)
        review_tensor = review_tensor.view(self.n_embed, -1)
        return review_tensor