import pandas as pd

from modules.data_preprocessing import preprocess_review

from hyperparameters import HyperParameters as params
from torchtext.vocab import GloVe
import torch

def create_model():
    """
    Trains, evaluates, and exports a deep, transformer-based model to
    classify textual movie reviews as positive or negative.

    The final model and evaluation graphs are exported to the `output/`
    directory.
    """

    # Training can be accomplished faster on a GPU, if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Device: {}'.format(device))

    # Step 1 - Load the data
    data_raw_df = pd.read_csv('data/imdb_dataset.csv')

    # Step 2 - Preprocess the data
    reviews = []
    sentiment = []
    sentiment_map= {
        'negative': 0,
        'positive': 1
    }
    
    vocab = set()
    n_reviews = 0
    for i, row in data_raw_df.iterrows():
        reviews.append(preprocess_review(row['review'], params.block_size))
        for word in reviews[-1]:
            vocab.add(word)
        sentiment.append(sentiment_map[row['sentiment']])
        n_reviews += 1
        if n_reviews == params.max_reviews:
            break

    print('Number of reviews: {}'.format(len(reviews)))
    print('Max review length (words): {}'.format(params.block_size))
    print('Vocab size: {}'.format(len(vocab)))
    print('Sample review: \n\t{}'.format(reviews[0]))

    # The movie reviews dataset is too small to train word emebddings,
    # so pretrained embeddings (GloVe) are used instead
    glove = GloVe(name="6B", dim=params.n_embed)

    x = glove.get_vecs_by_tokens(reviews[0], lower_case_backup=True)
    for i in range(1, len(reviews)):
        x = torch.cat((x, glove.get_vecs_by_tokens(reviews[i], lower_case_backup=True)))

    print(x.shape)

if __name__ == '__main__':
    create_model()