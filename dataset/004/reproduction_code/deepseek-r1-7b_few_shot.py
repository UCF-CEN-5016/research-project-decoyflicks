@staticmethod 
def _sample_from_beta(alpha, beta, shape): 
    sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)  
    sample_beta   = tf.random.gamma(shape, 1.0, beta=beta)
    return (sample_alpha/(sample_alpha + sample_beta), )

import tensorflow as tf

@staticmethod 
def _sample_from_beta(alpha, beta, shape): 
    sample_alpha = tf.random.gamma(shape, alpha, beta=1.0)  
    sample_beta   = tf.random.gamma(shape, beta, beta=1.0)
    return (tf.dtypes.cast(sample_alpha/(sample_alpha + sample_beta), dtype=tf.float32), )