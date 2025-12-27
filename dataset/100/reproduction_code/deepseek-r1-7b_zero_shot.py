class RotaryEmbedding:
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the first backward pass with retain_graph=True to avoid freeing intermediates after first call
        if hasattr(self, 'rospy') and not self.rospy:
            with torch.enable_grad():
                # Compute freqs without caching for the first time
                freqs = torch.arange(self.dim, device=x.device).float().view(1, -1)
                freqs = x * freqs.exp()
            
            # Cache the computed frequencies to avoid recomputing in subsequent forward passes
            self.cached_freqs = freqs
            self.rospy = True
            
        if not hasattr(self, 'rospy') or self.rospy:
            # Use cached frequencies for subsequent calls without recomputation
            return self.cached_freqs

        with torch.no_grad():
            # For the first forward pass (without caching), compute freqs as before
            freqs = torch.arange(self.dim, device=x.device).float().view(1, -1)
            freqs = x * freqs.exp()
        
        if not hasattr(self, 'rospy') or self.rospy:
            # Return cached frequencies to avoid redundant computation in subsequent forward calls
            return self.cached_freqs

        # Perform backward pass with retain_graph=True to keep intermediates for the second call
        loss = ...  # Compute loss as usual
        loss.backward(retain_graph=True)