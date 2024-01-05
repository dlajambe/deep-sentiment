class HyperParameters:
    """Stores hyperparameters that can be altered to change model
    architecture, training, and evaluation characteristics.
    """
    # Max reviews - can be reduced to reduce runtime
    max_reviews = 1000

    # Reviews longer than the block size are truncated to reduce
    # training time. However, this destroys information and can result
    # in worse model performance; when training a model for production,
    # the block size should be kept as high as possible.
    block_size = 100

    # The number of embeddings must be divisible by the number of heads
    n_embed = 50
    if n_embed not in [50, 100, 200, 300]:
        raise ValueError('GloVe embeddings must of dim 50, 100, 200, or 300')
