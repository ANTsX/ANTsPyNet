
_deterministic = False
_random_seed = None

def set_antsxnet_deterministic(on=True, seed_value=123):
    """
    Set deterministic behavior globally for the package.

    Parameters
    ----------
    on : bool
        Whether to enable deterministic mode.
    seed_value : int or None
        Random seed to use if deterministic mode is enabled.
    """
    global _deterministic, _random_seed
    _deterministic = on
    _random_seed = seed_value

    if _deterministic:
        import ants
        import random
        import tensorflow as tf

        ants.config.set_ants_deterministic(on=on, seed_value=seed_value) 
        tf.config.experimental.enable_op_determinism()
      
        if _random_seed is not None:

            np.random.seed(_random_seed)
            random.seed(_random_seed)
            tf.random.set_seed(_random_seed)
