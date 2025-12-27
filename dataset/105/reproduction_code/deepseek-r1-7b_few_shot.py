import torch

class ResidualVQ(torch.nn.Module):
    def __init__(self, embedding_dim, n_e, device):
        super().__init__()
        self.n_e = n_e
        self.embedding_dim = embedding_dim
        self.register_buffer('table', None)
        self.register_buffer('table_ema', None)
        self.register_buffer('last table update gradient', None)
        self.register_buffer('last exp moving average gradient', None)
        self.register_buffer('expired', None)
        self.expire_event hook = hooks.EventHook()
        self.table, self.table_ema, self.last table update gradient, self.last exp moving average gradient) in [self.expire_event_hook])
        self.device = device

    def forward(self, x):
        # ... (forward pass remains unchanged)

    def _replace(self, indices, mask, shape, value):
        """Replace entries in the codebook at positions `indices` with values from
        `value`. This is a helper method that allows for different ways to update
        the codebook, including through interpolation and nearest neighbor.
        """
        # Handle replacement when new codebook vectors are available
        if self.training:
            if shape == (self.n_e + 1,):
                # Expand the indices to match this new size
                indices = torch.cat((indices, indices.new_ones((indices.size(0), 1))), dim=1)
            
            # Ensure that `value` has sufficient elements for all positions in the mask
            if value.size(0) < (indices.size(0) * mask.size(1)):
                raise ValueError("Value tensor must have enough elements to fill all positions being updated.")
            
            # Replace entries using indices and mask
            self.table.data[indices] = value[mask]
        
        return super()._replace(indices, mask, shape, value)

    def update_codebook_withhook(self, indices, mask, value):
        if not self.expired:
            new_codebook = value[mask].chunk(self.n_e, dim=1)
            for i in range(self.n_e):
                codebook_part = new_codebook[i]
                codebook_part.data = codebook_part.data.t()
                
                # Update the codebook
                prev_codebook_part = self.table[:, :, i:i+1]
                loss = F.mse_loss(prev_codebook_part, codebook_part.data, reduction='none').mean(0).mean(1)
                
                # Exponential moving average update
                self.table_ema[:, :, i:i+1] = (1 - self.exp_momentum) * prev_codebook_part + \
                    self.exp_momentum * codebook_part.data
                
                # Gradient from EMA towards the parameter
                grad_ema = exp_momentum * (self.table_ema[:, :, i:i+1] - self.table[:, :, i:i+1])
                
                # Update the parameter in the opposite direction of the gradient
                if hasattr(self, 'last table update gradient'):
                    self.table.data.copy_(self.table.data - grad_ema - self.last table update gradient)
                else:
                    self.table.data = self.table.data - grad_ema
                
                # Track gradients for EMA update
                if hasattr(self, 'last exp moving average gradient'):
                    self.last exp moving average gradient = grad_ema
                else:
                    setattr(self, 'last exp moving average gradient', grad_ema)
        return super().update_codebook_withhook(indices, mask, value)

    def _replace_withouthook(self, indices, mask, shape, value):
        if not self.expired:
            # Expand the indices to match this new size
            indices = torch.cat((indices, indices.new_ones((indices.size(0), 1))), dim=1)
            
            # Ensure that `value` has sufficient elements for all positions in the mask
            if value.size(0) < (indices.size(0) * mask.size(1)):
                raise ValueError("Value tensor must have enough elements to fill all positions being updated.")
            
            # Replace entries using indices and mask
            self.table.data[indices] = value[mask]
        return super()._replace_withouthook(indices, mask, shape, value)

    def _replace_withhook(self, indices, mask, shape, value):
        if not self.expired:
            new_codebook = value[mask].chunk(self.n_e, dim=1)
            for i in range(self.n_e):
                codebook_part = new_codebook[i]
                codebook_part.data = codebook_part.data.t()
                
                # Update the codebook
                prev_codebook_part = self.table_ema[:, :, i:i+1]
                loss = F.mse_loss(prev_codebook_part, codebook_part.data, reduction='none').mean(0).mean(1)
                
                # Exponential moving average update
                self.table[:, :, i:i+1] = (1 - self.exp_momentum) * prev_codebook_part + \
                    self.exp_momentum * codebook_part.data
                
                # Gradient from EMA towards the parameter
                grad_ema = exp_momentum * (self.table_ema[:, :, i:i+1] - self.table[:, :, i:i+1])
                
                # Update the parameter in the opposite direction of the gradient
                if hasattr(self, 'last table update gradient'):
                    self.table.data.copy_(self.table.data - grad_ema - self.last table update gradient)
                else:
                    self.table.data = self.table.data - grad_ema
                
                # Track gradients for EMA update
                if hasattr(self, 'last exp moving average gradient'):
                    self.last exp moving average gradient = grad_ema
                else:
                    setattr(self, 'last exp moving average gradient', grad_ema)
        return super()._replace_withhook(indices, mask, shape, value)

    def _replace_withouthook(self, indices, mask, shape, value):
        if not self.expired:
            new_codebook = value[mask].chunk(self.n_e, dim=1)
            for i in range(self.n_e):
                codebook_part = new_codebook[i]
                codebook_part.data = codebook_part.data.t()
                
                # Update the codebook
                prev_codebook_part = self.table_ema[:, :, i:i+1]
                loss = F.mse_loss(prev_codebook_part, codebook_part.data, reduction='none').mean(0).mean(1)
                
                # Exponential moving average update
                self.table[:, :, i:i+1] = (1 - self.exp_momentum) * prev_codebook_part + \
                    self.exp_momentum * codebook_part.data
                
                # Gradient from EMA towards the parameter
                grad_ema = exp_momentum * (self.table_ema[:, :, i:i+1] - self.table[:, :, i:i+1])
                
                # Update the parameter in the opposite direction of the gradient
                if hasattr(self, 'last table update gradient'):
                    self.table.data.copy_(self.table.data - grad_ema - self.last table update gradient)
                else:
                    self.table.data = self.table.data - grad_ema
                
                # Track gradients for EMA update
                if hasattr(self, 'last exp moving average gradient'):
                    self.last exp moving average gradient = grad_ema
                else:
                    setattr(self, 'last exp moving average gradient', grad_ema)
        return super()._replace_withouthook(indices, mask, shape, value)

    def _replace(self, indices, mask, shape, value):
        if not self.expired:
            # Expand the indices to match this new size
            if isinstance(indices, (list, tuple)):
                indices = torch.stack(indices, dim=0)
            if len(indices.size()) == 1:
                # Convert 1D index to 2D with an extra dimension for broadcasting
                indices = indices.view(-1, 1)

            # Ensure that `value` has sufficient elements for all positions in the mask
            if value.size(0) < (indices.size(0) * mask.size(1)):
                raise ValueError("Value tensor must have enough elements to fill all positions being updated.")

            # Replace entries using indices and mask
            self.table.data[indices] = value[mask]
        return super()._replace(indices, mask, shape, value)

    def update_codebook_withhook(self, indices, mask, value):
        if not self.expired:
            new_codebook = value[mask].chunk(self.n_e, dim=1)
            for i in range(self.n_e):
                codebook_part = new_codebook[i]
                codebook_part.data = codebook_part.data.t()
                
                # Update the codebook
                prev_codebook_part = self.table_ema[:, :, i:i+1]
                loss = F.mse_loss(prev_codebook_part, codebook_part.data, reduction='none').mean(0).mean(1)
                
                # Exponential moving average update
                self.table[:, :, i:i+1] = (1 - self.exp_momentum) * prev_codebook_part + \
                    self.exp_momentum * codebook_part.data
                
                # Gradient from EMA towards the parameter
                grad_ema = exp_momentum * (self.table_ema[:, :, i:i+1] - self.table[:, :, i:i+1])
                
                # Update the parameter in the opposite direction of the gradient
                if hasattr(self, 'last table update gradient'):
                    self.table.data.copy_(self.table.data - grad_ema - self.last table update gradient)
                else:
                    self.table.data = self.table.data - grad_ema
                
                # Track gradients for EMA update
                if hasattr(self, 'last exp moving average gradient'):
                    self.last exp_momentum * (self.table_ema[:, :, i:i+1] - self.table[:, :, i:i+1])
                else:
                    setattr(self, 'last exp moving average gradient', grad_ema)
        return super().update_codebook_withhook(indices, mask, value)

    def _replace_withouthook(self, indices, mask, shape, value):
        if not self.expired:
            new_codebook = value[mask].chunk(self.n_e, dim=1)
            for i in range(self.n_e):
                codebook_part = new_codebook[i]
                codebook_part.data = codebook_part.data.t()
                
                # Update the codebook
                prev_codebook_part = self.table_ema[:, :, i:i+1]
                loss = F.mse_loss(prev_codebook_part, codebook_part.data, reduction='none').mean(0).mean(1)
                
                # Exponential moving average update
                self.table[:, :, i:i+1] = (1 - self.exp_momentum) * prev_codebook_part + \
                    self.exp_momentum * codebook_part.data
                
                # Gradient from EMA towards the parameter
                grad_ema = exp_momentum * (self.table_ema[:, :, i:i+1] - self.table[:, :, i:i+1])
                
                # Update the parameter in the opposite direction of the gradient
                if hasattr(self, 'last table update gradient'):
                    self.table.data.copy_(self.table.data - grad_ema - self.last table update gradient)
                else:
                    self.table.data = self.table.data - grad_ema
                
                # Track gradients for EMA update
                if hasattr(self, 'last exp moving average gradient'):
                    self.last exp_momentum * (self.table_ema[:, :, i:i+1] - self.table[:, :, i:i+1])
                else:
                    setattr(self, 'last exp moving average gradient', grad_ema)
        return super()._replace_withouthook(indices, mask, shape, value)

    def _replace(self, indices, mask, shape, value):
        if not self.expired:
            # Expand the indices to match this new size
            if isinstance(indices, (list, tuple)):
                indices = torch.stack(indices, dim=0)
            if len(indices.size()) == 1:
                # Convert 1D index to 2D with an extra dimension for broadcasting
                indices = indices.view(-1, 1)

            # Ensure that `value` has sufficient elements for all positions in the mask
            if value.size(0) < (indices.size(0) * mask.size(1)):
                raise ValueError("Value tensor must have enough elements to fill all positions being updated.")

            # Replace entries using indices and mask
            self.table.data[indices] = value[mask]
        return super()._replace(indices, mask, shape, value)