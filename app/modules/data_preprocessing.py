from string import punctuation

def preprocess_review(review_raw: str, max_length_words: int) -> list:
    """Preprocesses a review string by removing punctuation, separating
    into a list of words, and truncating the review if it exceeds the
    maximum review length threshold.

    Parameters
    ----------
    review_raw : str
        The review to be preprocessed.

    max_review_length : int
        The maximum allowable number of words in the preprocessed 
        review. If the preprocessed review exceeds this length, it is
        truncated.

    Returns
    -------
    review_new : list
        A list of strings (words) reprepsenting the final, preprocessed
        review.
    """
    review_new = review_raw.replace('<br />', ' ')
    review_new = ''.join(
        [char for char in review_new if char not in punctuation])
    review_new = review_new.split()[:max_length_words]
    return review_new