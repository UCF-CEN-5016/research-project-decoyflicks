class GaussianDiffusion:
    def ddim_sample(self, img, time_cond, self_cond):
        self.model_predictions(img, time_cond, self_cond, clip_x_start=True)

    def model_predictions(self, x, t, clip_x_start=False):
        pass

class LearnedGaussianDiffusion(GaussianDiffusion):
    def model_predictions(self, x, t, clip_x_start=False):
        super().model_predictions(x, t, clip_x_start=clip_x_start)

# Example usage
diffusion = LearnedGaussianDiffusion()
diffusion.ddim_sample(img="image", time_cond="time", self_cond="self_cond")