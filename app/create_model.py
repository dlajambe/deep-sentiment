import os

import pandas as pd

from modules.data_preprocessing import tokenize_review, TokenDataset

from hyperparameters import HyperParameters as params
import torch
from torch.utils.data import DataLoader
import time
from modules.sentiment_nn import SentimentNet, train_model, calc_metrics

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

    # Step 3 - Data partitioning
    # A training, validation, and testing partion are required to train
    # the model's parameters, determine when to stop training, and
    # evaluate the final model's performance, respectively
    indices = [i for i in range(len(reviews))]
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
    model = SentimentNet(params.n_embed, params.block_size, device).to(device)
    print('Beginning model training')
    dataset_train = TokenDataset(
        [reviews[idx] for idx in train],
        [sentiment[idx] for idx in train],
        params.n_embed)
    dataset_val = TokenDataset(
        [reviews[idx] for idx in val], 
        [sentiment[idx] for idx in val],
        params.n_embed)
    train_model(
        model, 
        dataset_train, dataset_val,
        params.max_epochs, params.batch_size, params.lr, device)
    
    # Step 5 - Estimate the model's performance using the testing data
    dataset_test = TokenDataset(
        [reviews[idx] for idx in test], 
        [sentiment[idx] for idx in test],
        params.n_embed)
    data_loader_test = DataLoader(
        dataset_test,
        shuffle=True,
        drop_last=True,
        batch_size=params.batch_size)
    
    model.eval()
    loss_test, accuracy_test = calc_metrics(model, data_loader_test, device)
    print('Final model performance metrics:')
    print('\tLoss (test): {0}\tAccuracy (test): {1}'.format(
        round(loss_test, 4), round(accuracy_test, 4))
    )
    end_time = time.perf_counter()
    print('Total script runtime: {} seconds'.format(end_time - start_time))

    # Step 6 - Export the model
    output_dir = 'output/'
    if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

    torch.save(model.state_dict(), output_dir + 'sentiment_transformer.pth')


if __name__ == '__main__':
    create_model()