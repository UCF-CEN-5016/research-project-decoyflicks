import torch
from torch import nn, autocast
from einops import rearrange, reduce, get_at
from einops.layers.torch import Reduce

# Assuming these imports are available in the environment where this code runs
# from vector_quantize_pytorch import LFQ  # Uncomment if needed
# from utils import exists, pack, F, unpack, round_up_multiple, random, get_maybe_sync_seed, partial  # Replace with actual imports

class ResidualLFQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(self, *, dim, num_quantizers, codebook_size, quantize_dropout=False, quantize_dropout_cutoff_index=0, quantize_dropout_multiple_of=1, soft_clamp_input_value=None, **kwargs):
        super().__init__()
        from math import log2  # Import log2 here as it was undefined
        codebook_dim = int(log2(codebook_size))

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection

        self.num_quantizers = num_quantizers

        self.layers = nn.ModuleList([])

        for ind in range(num_quantizers):
            codebook_scale = 2 ** -ind

            # Assuming LFQ is available and correctly imported
            lfq = LFQ(dim=codebook_dim, codebook_scale=codebook_scale, soft_clamp_input_value=soft_clamp_input_value, **kwargs)

            self.layers.append(lfq)

            if exists(soft_clamp_input_value):
                soft_clamp_input_value *= 0.5

        assert all([not lfq.has_projections for lfq in self.layers])
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim=0)
        return codebooks

    def get_codes_from_indices(self, indices):
        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        indices, ps = pack([indices], 'b * q')
        if quantize_dim < self.num_quantizers:
            assert self.quantize_dropout > 0., 'quantize dropout must be greater than 0 if you wish to reconstruct from a signal with less fine quantizations'
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value=-1)
        mask = indices == -1.
        indices = indices.masked_fill(mask, 0)
        all_codes = get_at('q [c] d, b n q -> q b n d', self.codebooks, indices)
        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)
        all_codes, = unpack(all_codes, ps, 'q b * d')
        return all_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def forward(self, x, mask=None, return_all_codes=False, rand_quantize_dropout_fixed_seed=None):
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device
        x = self.project_in(x)
        quantized_out = 0.
        residual = x
        all_losses = []
        all_indices = []
        should_quantize_dropout = self.training and self.quantize_dropout
        if should_quantize_dropout:
            if not exists(rand_quantize_dropout_fixed_seed):
                rand_quantize_dropout_fixed_seed = get_maybe_sync_seed(device)
            rand = random.Random(rand_quantize_dropout_fixed_seed)
            rand_quantize_dropout_index = rand.randrange(self.quantize_dropout_cutoff_index, num_quant)
            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1
            null_indices = torch.full(x.shape[:2], -1., device=device, dtype=torch.long)  # Assign here before usage
            null_loss = torch.tensor(0., device=device, dtype=x.dtype)  # Assign here before usage
        with autocast('cuda', enabled=False):
            for quantizer_index, layer in enumerate(self.layers):
                if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                    all_indices.append(null_indices)
                    all_losses.append(null_loss)
                    continue
                quantized, indices, loss = layer(residual, mask=mask)
                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized
                all_indices.append(indices)
                all_losses.append(loss)
        quantized_out = self.project_out(quantized_out)
        all_losses, all_indices = map(partial(torch.stack, dim=-1), (all_losses, all_indices))
        ret = (quantized_out, all_indices, all_losses)
        if not return_all_codes:
            return ret
        all_codes = self.get_codes_from_indices(all_indices)
        return (*ret, all_codes)