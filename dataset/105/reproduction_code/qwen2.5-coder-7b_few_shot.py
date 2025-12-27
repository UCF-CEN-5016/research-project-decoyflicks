import torch
from typing import Tuple


class ResidualVQ(torch.nn.Module):
    def __init__(self, codebook_size: int, dim: int):
        """
        Simple residual vector-quantization-like module holding a learnable codebook.

        Args:
            codebook_size: Number of codebook entries.
            dim: Dimensionality of each codebook vector.
        """
        super().__init__()
        self._num_codes = int(codebook_size)
        self._code_dim = int(dim)
        # Parameter storing the codebook vectors
        self.embeddings = torch.nn.Parameter(torch.randn(self._num_codes, self._code_dim))

    @property
    def num_codes(self) -> int:
        return self._num_codes

    @property
    def code_dim(self) -> int:
        return self._code_dim

    def _validate_replace_inputs(self, samples: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int]:
        """
        Validate the inputs for the replace operation.

        Returns:
            A tuple (batch_size, feature_dim)
        """
        if not isinstance(samples, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("samples and mask must be torch.Tensor instances")
        if samples.dim() != 2:
            raise ValueError("samples must be a 2D tensor of shape (batch, dim)")
        if mask.dim() != 1:
            raise ValueError("mask must be a 1D tensor of shape (batch,)")
        batch_size, feat_dim = samples.size()
        if batch_size != mask.size(0):
            raise ValueError("Sampled data size does not match mask size")
        if feat_dim != self._code_dim:
            raise ValueError("Sample feature dimension does not match codebook dimension")
        return batch_size, feat_dim

    def replace(self, sampled: torch.Tensor, mask: torch.Tensor) -> None:
        """
        Replace entries in the codebook with provided sampled vectors at random indices,
        but only for positions where mask is True.

        Args:
            sampled: Tensor of shape (batch, dim) containing candidate replacements.
            mask: 1D boolean tensor of shape (batch,) indicating which sampled rows to use.
        """
        batch_size, _ = self._validate_replace_inputs(sampled, mask)

        # Randomly select one codebook index per sampled row
        indices = torch.randint(0, self.embeddings.size(0), (batch_size,))

        # Ensure boolean mask
        boolean_mask = mask.type(torch.bool)

        # Perform the in-place update on the underlying data tensor to preserve semantics
        self.embeddings.data[indices][boolean_mask] = sampled


# Test the code
if __name__ == "__main__":
    model = ResidualVQ(codebook_size=10000, dim=512)
    x = torch.randn(9330, 512)
    mask = torch.rand(9330, dtype=torch.bool)
    model.replace(x, mask)