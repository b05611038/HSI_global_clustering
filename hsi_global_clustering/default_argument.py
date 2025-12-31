# Default arguments for training scripts

DEFAULT_MODEL_KWARGS = {
    'num_bands': 301,
    'encoder_kwargs': {
        'n_spectral_layers': 3,
        'spectral_kernel_size': 9,
        'embed_dim': 32,
        'bias': False,
    },
    'mean_shift_kwargs': {
        'embed_dim': 32,
        'n_clusters': 4,
        'num_iters': 5,
    },
    'loss_weights': {
        'orth': 0.001,
        'bal': 2.0,
        'unif': 2.0,
        'cons': 1.0,
    },
}

DEFAULT_OPTIMIZER_KWARGS = {
    'lr': 1e-4,
    'betas': (0.9, 0.999),
    'weight_decay': 0.01,
}

DEFAULT_LOSS_WEIGHT_SCHEDULING = {
    'lambda_unif': False,
    'lambda_orth': False,
    'lambda_bal': False,
    'lambda_cons': False,
}

DEFAULT_EMA_DECAY = 0.99
DEFAULT_EMA_KICK = 0.05
DEFAULT_EMA_KICK_SCHEDULING = False
