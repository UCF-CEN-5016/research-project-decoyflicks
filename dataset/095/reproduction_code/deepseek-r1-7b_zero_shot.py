def ddim_sample(self, x, t):
    """Generate samples using DDIM with learned denoising."""
    # ... existing code ...
    
    for i in range(len(t) - 1, -1, -1):
        time_step = self._diffusion.p diffusion_model(t[i])
        # ... (code continues)
        pred_noise, x_start = self.model_predict(
            x, t[i], clip_x_start=False
        )
        # ... rest of the code ...