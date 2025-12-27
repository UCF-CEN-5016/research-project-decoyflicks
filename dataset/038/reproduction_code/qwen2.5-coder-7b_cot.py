class RotaryPositionalEmbeddingTransformer:
    def __init__(self, cos_cached, sin_cached, d=None, pos_wt=None, analyzer=None):
        """
        cos_cached, sin_cached: cached cosine/sine tensors (NumPy or PyTorch style)
        d: optional depth limit used when cached tensors have extra dims
        pos_wt: optional callable to apply to the pre-rotated tensor (matches earlier usage)
        analyzer: optional callable to pre-process input x (defaults to identity)
        """
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached
        self.d = d
        self.pos_wt = pos_wt
        self.analyzer = analyzer or (lambda t: t)

    def _ndim(self, tensor):
        try:
            return tensor.ndim
        except AttributeError:
            return len(tensor.shape)

    def _slice_for_batch(self, tensor, batch_size):
        """
        Choose an appropriate slice of `tensor` for the given batch_size.
        If the cached tensor has 4 or more dimensions, include the final depth slice self.d if provided.
        Otherwise, slice along the leading batch dimension.
        """
        n_dims = self._ndim(tensor)
        if n_dims >= 4 and self.d is not None:
            # e.g. tensor shape: (batch, heads, seq, channels)
            return tensor[:batch_size, ..., :self.d]
        else:
            return tensor[:batch_size]

    def apply_rope(self, x, neg_half_x):
        """
        Apply rotary positional embedding operation:
        x_rope = analyze(x) [optionally pos_wt applied]
        rope_out = x_rope * cos_slice + neg_half_x * sin_slice

        Returns rope_out.
        """
        batch_size = x.shape[0]
        processed = self.analyzer(x)

        # If a positional-weighting function was provided in earlier code paths,
        # apply it to the pre-rotated tensor to preserve compatibility.
        if self.pos_wt is not None:
            processed = self.pos_wt(processed)

        cos_slice = self._slice_for_batch(self.cos_cached, batch_size)
        sin_slice = self._slice_for_batch(self.sin_cached, batch_size)

        rope_out = processed * cos_slice + neg_half_x * sin_slice
        return rope_out


# Example usage (kept here for completeness; adapt or remove as needed):
# transformer = RotaryPositionalEmbeddingTransformer(cos_cached, sin_cached, d=self.d, pos_wt=pos_wt, analyzer=analyze)
# rope_out = transformer.apply_rope(x, neg_half_x)