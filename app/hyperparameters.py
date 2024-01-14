class HyperParameters:
    """Stores hyperparameters that can be altered to change model
    architecture, training, and evaluation characteristics.
    """
    # Used to ensure deterministic results during random batch selection
    seed = 1337

    # Determines the fraction of the data that are assigned (randomly)
    # to each data partition
    partitions_fracs = {
        'train': 0.6,
        'val': 0.2,
        'test': 0.2
    }

    # Max reviews - can be reduced to reduce runtime
    max_reviews = 50000

    # Reviews longer than the block size are truncated to reduce
    # training time. However, this destroys information and can result
    # in worse model performance; when training a model for production,
    # the block size should be kept as high as possible.
    block_size = 400

    # The number of embeddings must be divisible by the number of heads
    n_embed = 200
    n_heads = 8
    if n_embed not in [50, 100, 200, 300]:
        raise ValueError('GloVe embeddings must of dim 50, 100, 200, or 300')
    elif n_embed % n_heads != 0:
        raise ValueError(
            'n_embed must be divisible n_heads')
    
    # Training parameters
    
    # The number of reviews that should be included in the gradient
    # calculation for a single parameter update using backpropagation
    batch_size = 64

    # Controls what fraction of nodes are randomly deactivated during
    # training
    dropout_frac = 0.3

    # The total number of passes through the data during model training
    max_epochs = 50

    # The learning rate
    lr = 1e-3

    def print():
        print('Hyperparameters:')
        print('\tSeed: {}'.format(HyperParameters.seed))
        print('\tMax reviews: {}'.format(HyperParameters.max_reviews))
        print('\tBlock size: {}'.format(HyperParameters.block_size))
        print('\tEmbedding dim: {}'.format(HyperParameters.n_embed))
        print('\tNumber of heads: {}'.format(HyperParameters.n_heads))
        print('\tBatch size: {}'.format(HyperParameters.batch_size))
        print('\tMax epochs: {}'.format(HyperParameters.max_epochs))
        print('\tLearning rate: {}'.format(HyperParameters.lr))

    
