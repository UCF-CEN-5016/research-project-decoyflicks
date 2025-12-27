import torch
from typing import Tuple


class Codebook:
    """
    Simple codebook / embedding container that supports sampling neighbor codes.
    This class intentionally keeps the core logic straightforward and mirrors
    typical behaviors required by ResidualVQ without changing functionality.
    """

    def __init__(self, num_codes: int, dim: int, device: torch.device = None):
        self.num_codes = num_codes
        self.dim = dim
        self.device = device or torch.device("cpu")
        # embeddings stored as a parameter-like tensor (no grad by default here)
        self.embeddings = torch.randn(num_codes, dim, device=self.device)

    def sample_from_neighbors(
        self, indices: torch.LongTensor, n_neighbors: int
    ) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        """
        Given a 1D tensor `indices` (shape: [N]), sample `n_neighbors` neighbor codes
        for each index and return:
          - sampled: tensor of shape [N * n_neighbors, dim]
          - mask: boolean tensor of shape [N * n_neighbors] indicating valid samples
          - neighbor_indices: long tensor of shape [N * n_neighbors] with the actual indices used

        The default neighbor strategy here is deterministic offsets (for reproducibility).
        """
        if n_neighbors <= 0:
            # Return empty tensors consistent in shape
            sampled = torch.empty(0, self.dim, device=self.device)
            mask = torch.empty(0, dtype=torch.bool, device=self.device)
            neighbor_indices = torch.empty(0, dtype=torch.long, device=self.device)
            return sampled, mask, neighbor_indices

        # Ensure input indices are 1D long tensor
        indices_1d = indices.view(-1).long().to(self.device)
        n_items = indices_1d.size(0)

        # For each index, create neighbor indices by adding offsets and wrapping around
        offsets = torch.arange(1, n_neighbors + 1, device=self.device).unsqueeze(0)  # [1, K]
        base = indices_1d.unsqueeze(1)  # [N, 1]
        neighbor_matrix = (base + offsets) % self.num_codes  # [N, K]
        neighbor_flat = neighbor_matrix.reshape(-1)  # [N*K]

        # For sampling semantics, we take embeddings for these neighbor indices
        sampled = self.embeddings[neighbor_flat]  # [N*K, dim]

        # Construct a mask indicating validity of each sample.
        # Here we mark all as valid (True) but preserve shape and dtype expected.
        mask = torch.ones(neighbor_flat.size(0), dtype=torch.bool, device=self.device)

        return sampled, mask, neighbor_flat


class ResidualVQ:
    """
    Residual Vector Quantizer that uses a Codebook and performs operations
    which may write sampled codes back into the codebook embeddings.
    The implementation preserves original behavior but improves indexing clarity
    and shape management to avoid subtle chained-indexing shape mismatches.
    """

    def __init__(self, codebook: Codebook):
        self.codebook = codebook

    def _expand_indices_for_neighbors(
        self, base_indices: torch.LongTensor, n_neighbors: int
    ) -> torch.LongTensor:
        """
        Given base_indices (shape [N]) and n_neighbors K,
        returns a flat tensor of shape [N*K] where each base index is repeated K times.
        """
        if n_neighbors <= 0:
            return torch.empty(0, dtype=torch.long, device=base_indices.device)

        base = base_indices.view(-1).long()
        repeated = base.unsqueeze(1).expand(-1, n_neighbors).contiguous().view(-1)
        return repeated

    def assign_samples_to_codebook(
        self,
        base_indices: torch.LongTensor,
        n_neighbors: int,
        sampled: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> None:
        """
        Assign `sampled` vectors into codebook embeddings at positions indicated by
        base_indices repeated for each neighbor, then masked.

        This replaces ambiguous chained indexing like `self.embeddings[base_indices][mask] = sampled`
        with unambiguous advanced indexing that flattens indices first and then applies mask,
        ensuring shapes match exactly.
        """
        device = self.codebook.embeddings.device
        base_indices = base_indices.to(device).view(-1).long()

        # Build the expanded row indices corresponding to each neighbor sample.
        expanded_rows = self._expand_indices_for_neighbors(base_indices, n_neighbors)
        # `mask` is expected to be of shape [N * n_neighbors]; flatten it to be safe.
        mask_flat = mask.view(-1).to(device)

        if expanded_rows.numel() != mask_flat.numel():
            # Defensive check: shapes must align. If not, raise with clear info.
            raise RuntimeError(
                f"Indexing mismatch: expanded rows ({expanded_rows.numel()}) != mask elements ({mask_flat.numel()})"
            )

        # Select the actual row indices that will be written to (apply mask).
        target_rows = expanded_rows[mask_flat]

        if target_rows.numel() != sampled.size(0):
            # Another defensive check: number of target rows must equal number of sampled entries.
            raise RuntimeError(
                f"shape mismatch: sampled tensor of shape {tuple(sampled.shape)} "
                f"cannot be assigned to indexing result of shape ({target_rows.numel()}, {self.codebook.dim})"
            )

        # Now perform the assignment using advanced indexing (single indexed dimension).
        # Use .data to mimic in-place update semantics on embeddings storage (consistent with original behavior).
        self.codebook.embeddings.data[target_rows] = sampled

    def process_and_update(
        self,
        token_indices: torch.LongTensor,
        n_neighbors: int,
    ) -> None:
        """
        High-level method that samples neighbors for token_indices and then writes
        the sampled neighbor embeddings back into the codebook using appropriate indexing.

        This preserves original functional flow: get sampled neighbors and mask, then assign.
        """
        sampled, mask, neighbor_indices = self.codebook.sample_from_neighbors(
            token_indices, n_neighbors
        )

        # When there are no neighbors, nothing to assign.
        if sampled.numel() == 0:
            return

        # The neighbor sampling method returns a flat list of neighbor indices (neighbor_indices)
        # and sampled embeddings in the same flattened order. But the assignment expects
        # base_indices repeated per neighbor and a mask of the same flat size. We'll compute
        # those derived indices and call assign_samples_to_codebook to keep logic clear.
        self.assign_samples_to_codebook(token_indices, n_neighbors, sampled, mask)


# Example usage (for integration/testing purposes only; no behavior change expected):
if __name__ == "__main__":
    device = torch.device("cpu")
    cb = Codebook(num_codes=10000, dim=512, device=device)
    rvq = ResidualVQ(cb)

    # Suppose we have a batch of token indices (N tokens)
    tokens = torch.randint(0, cb.num_codes, (50,), dtype=torch.long, device=device)
    K = 3

    # Perform neighbor sampling and assignment back into the codebook
    rvq.process_and_update(tokens, K)