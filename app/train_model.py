import pandas as pd

from modules.data_preprocessing import preprocess_review

def main():
    # Step 1 - Load the data
    data_raw_df = pd.read_csv('data/imdb_dataset.csv')

    # Step 2 - Preprocess the data
    reviews = []
    sentiment = []
    sentiment_map= {
        'negative': 0,
        'positive': 1
    }
    
    # The maximum review length is used to filter out long reviews, 
    # thereby reducing the sequence length and training time. However,
    # this destroys data and can result in worse model performance; when
    # training a model for production, the maximum review length should
    # be kept as high as possible.
    max_review_length = 100
    vocab = set()
    for i, row in data_raw_df.iterrows():
        reviews.append(preprocess_review(row['review'], max_review_length))
        for word in reviews[-1]:
            vocab.add(word)
        sentiment.append(sentiment_map[row['sentiment']])

    print('Number of reviews: {}'.format(len(reviews)))
    print('Max review length (words): {}'.format(max_review_length))
    print('Vocab size: {}'.format(len(vocab)))
    print('Sample review: \n\t{}'.format(reviews[0]))

if __name__ == '__main__':
    main()