import torch
import torchaudio

AUDIO_FILEPATH = "audio.wav"
TEXT_FILEPATH = "text.txt"


def load_audio(path: str):
    """
    Load an audio file using torchaudio.

    Returns:
        waveform: Tensor of shape (channels, samples)
        sample_rate: int
    """
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate


def compute_mel_features(waveform: torch.Tensor, sample_rate: int, n_mels: int = 40, n_fft: int = 256):
    """
    Compute a Mel spectrogram and return it in shape (time, n_mels).

    Args:
        waveform: Tensor of shape (channels, samples)
        sample_rate: Sampling rate of the waveform
        n_mels: Number of Mel frequency bands
        n_fft: FFT window size

    Returns:
        mel_features: Tensor of shape (time, n_mels)
    """
    # Convert to mono if necessary
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
    )

    # mel_spec has shape (channels, n_mels, time)
    mel_spec = mel_transform(waveform)

    # Reduce channel dimension and return (time, n_mels) which is often expected by models
    mel_features = mel_spec.squeeze(0).transpose(0, 1)
    return mel_features


def ensure_batch_dimension(log_probs: torch.Tensor):
    """
    Ensure log_probs has a batch dimension. Acceptable input shapes:
      - (time, n_features) -> will become (1, time, n_features)
      - (batch, time, n_features) -> returns as-is

    Args:
        log_probs: Tensor with 2 or 3 dimensions

    Returns:
        Tensor with 3 dimensions (batch, time, n_features)
    """
    if not isinstance(log_probs, torch.Tensor):
        raise TypeError("log_probs must be a torch.Tensor")

    if log_probs.dim() == 2:
        return log_probs.unsqueeze(0)
    if log_probs.dim() == 3:
        return log_probs
    raise ValueError("log_probs must have 2 or 3 dimensions (time, features) or (batch, time, features)")


def get_model_log_probs(features: torch.Tensor):
    """
    Placeholder for model inference that produces log probabilities from features.

    Replace the body of this function with actual model inference code. Expected output shape:
      - (time, n_features) or (batch, time, n_features)

    For now this raises NotImplementedError to signal the required replacement.
    """
    raise NotImplementedError("Replace get_model_log_probs with model inference returning a tensor of shape "
                              "(time, n_features) or (batch, time, n_features)")


if __name__ == "__main__":
    waveform, sample_rate = load_audio(AUDIO_FILEPATH)

    window_size = 256
    n_mels = 40

    mel_features = compute_mel_features(waveform, sample_rate, n_mels=n_mels, n_fft=window_size)

    # Obtain log_probs from your model. The following call is a placeholder.
    # log_probs = get_model_log_probs(mel_features)
    #
    # After obtaining log_probs, ensure it has a batch dimension:
    # log_probs_batched = ensure_batch_dimension(log_probs)
    #
    # Proceed with alignment using log_probs_batched.
    pass