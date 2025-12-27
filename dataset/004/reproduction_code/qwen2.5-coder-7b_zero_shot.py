import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow_models.vision.ops import MixupAndCutmix


def sample_from_tf_beta(alpha: float, beta: float, count: int) -> np.ndarray:
    """Use MixupAndCutmix._sample_from_beta to draw samples and return a NumPy array."""
    samples = MixupAndCutmix._sample_from_beta(alpha, beta, (count,))
    # Convert TensorFlow tensor to NumPy if needed
    if hasattr(samples, "numpy"):
        return samples.numpy()
    return np.asarray(samples)


def sample_from_numpy_beta(alpha: float, beta: float, count: int) -> np.ndarray:
    """Draw samples using NumPy's beta distribution."""
    return np.random.beta(alpha, beta, count)


def plot_kde_comparison(samples_a: np.ndarray, samples_b: np.ndarray, clip: tuple = (0, 1)) -> None:
    """Plot KDEs of two sample arrays for visual comparison."""
    plt.figure()
    sns.kdeplot(samples_a, clip=clip, label="MixupAndCutmix._sample_from_beta")
    sns.kdeplot(samples_b, clip=clip, label="np.random.beta")
    plt.legend()
    plt.show()


def main() -> None:
    alpha = 0.2
    beta = 0.2
    sample_size = 100000

    tf_samples = sample_from_tf_beta(alpha, beta, sample_size)
    np_samples = sample_from_numpy_beta(alpha, beta, sample_size)

    plot_kde_comparison(tf_samples, np_samples, clip=(0, 1))


if __name__ == "__main__":
    main()