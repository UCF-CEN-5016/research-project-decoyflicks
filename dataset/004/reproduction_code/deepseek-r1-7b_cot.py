import tensorflow as tf
from tensorflow_probability import distributions as tfd

def _sample_from_beta(alpha, beta):
    # Generate Gamma samples with concentration=alpha and rate=1.0
    gamma_alpha = tfd.Gamma(concentration=alpha, rate=tf.constant(1.0))
    sample_alpha = gamma_alpha.sample()
    
    # Similarly for beta to generate gamma_beta
    gamma_beta = tfd.Gamma(concentration=beta, rate=tf.constant(1.0))
    sample_beta = gamma_beta.sample()
    
    # Calculate the Beta distributed samples
    z = sample_alpha / (sample_alpha + sample_beta)
    return z