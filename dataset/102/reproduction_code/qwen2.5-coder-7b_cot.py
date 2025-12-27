from torch import nn, Tensor


class SimpleModel(nn.Module):
    """A minimal model that conditionally applies dropout to the input."""

    def __init__(self, dropout_prob: float = 0.5) -> None:
        super(SimpleModel, self).__init__()
        self._dropout = self._build_dropout(dropout_prob)

    def _build_dropout(self, p: float) -> nn.Dropout:
        return nn.Dropout(p=p)

    def _apply_dropout(self, tensor: Tensor, enabled: bool) -> Tensor:
        if enabled:
            return self._dropout(tensor)
        return tensor

    def forward(self, input_tensor: Tensor, is_training: bool = True) -> Tensor:
        """
        Forward pass.

        Args:
            input_tensor: Input tensor to potentially apply dropout to.
            is_training: If True, apply dropout; otherwise return input unchanged.

        Returns:
            Tensor after optional dropout.
        """
        return self._apply_dropout(input_tensor, is_training)