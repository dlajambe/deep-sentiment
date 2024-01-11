import pandas as pd

from modules.data_preprocessing import tokenize_review

from hyperparameters import HyperParameters as params
from torchtext.vocab import GloVe
import torch
import time
from modules.sentiment_nn import SentimentNet, train_model

import random
import math

def create_model():
    """
    Trains, evaluates, and exports a deep, transformer-based model to
    classify textual movie reviews as positive or negative.

    The final model and evaluation graphs are exported to the `output/`
    directory.
    """

    # Training can be accomplished faster on a GPU, if available
    start_time = time.perf_counter()
    torch.manual_seed(params.seed)
    random.seed(params.seed)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = 'cpu'
    print('Device: {}'.format(device))

    # Step 1 - Load the data
    data_raw_df = pd.read_csv('data/imdb_dataset.csv')

    # Step 2 - Preprocess the review data
    reviews = []
    sentiment_map= {
        'negative': 0,
        'positive': 1
    }

    sentiment = []
    vocab = set()
    n_reviews = 0
    for i, row in data_raw_df.iterrows():
        reviews.append(tokenize_review(row['review'], params.block_size))
        for word in reviews[-1]:
            vocab.add(word)
        sentiment.append(sentiment_map[row['sentiment']])
        n_reviews += 1
        if n_reviews == params.max_reviews:
            break

    print('Number of reviews: {}'.format(len(reviews)))
    print('Max review length (words): {}'.format(params.block_size))
    print('Vocab size: {}'.format(len(vocab)))
    print('Sample review: \n\t{}'.format(reviews[9]))

    # Step 3 - Convert the data to Tensor format

    # The movie reviews dataset is too small to train word emebddings,
    # so pretrained embeddings (GloVe) are used instead
    glove = GloVe(name="6B", dim=params.n_embed)

    # Since static, pre-trained embeddings are being used, the reviews
    # can be vectorized in advance instead of being vectorized on the
    # fly during training, saving a significant amount of time. However,
    # doing this does cost significantly more memory.
    y = torch.zeros(size=(len(reviews),))
    x = torch.zeros(size=(len(reviews), params.block_size, params.n_embed))
    for i in range(len(reviews)):
        x[i] = glove.get_vecs_by_tokens(reviews[i])
        y[i] = sentiment[i]

    # Step 4 - Data partitioning
    # A training, validation, and testing partion are required to train
    # the model's parameters, determine when to stop training, and
    # evaluate the final model's performance, respectively
    indices = [i for i in range(len(x))]
    random.shuffle(indices)
    val_start = math.ceil(params.partitions_fracs['train']*len(indices))
    test_start = val_start + math.ceil(
        params.partitions_fracs['val']*len(indices))
    train = indices[:val_start]
    val = indices[val_start:test_start]
    test = indices[test_start:]
    print('Data partition fractions:')
    print('\tTraining: {}\t'.format(len(train)/len(indices)))
    print('\tValidation: {}\t'.format(len(val)/len(indices)))
    print('\tTesting: {}\t'.format(len(test)/len(indices)))

   
    # Step 4 - Initialize and train the model
    x = x.to(device)
    y = y.to(device)
    model = SentimentNet(params.n_embed, params.block_size).to(device)
    print('Beginning model training')
    train_model(
        model, 
        x[train], y[train], x[val], y[val],
        params.max_epochs, params.batch_size, params.lr)
    
    end_time = time.perf_counter()
    print('Total script runtime: {} seconds'.format(end_time - start_time))


if __name__ == '__main__':
    create_model()