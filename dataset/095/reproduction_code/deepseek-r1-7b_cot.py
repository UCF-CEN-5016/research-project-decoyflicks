self.model_predictions(img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True)

# Inside ddpm.py's ddim_sample function

if self.__class__.__name__ == 'DDPM':
    # Gaussian model: pass all parameters including clip_x_start and rederive_pred_noise
    call = self.gaussian_model.predictions
    call(x, t, clip_x_start=..., rederive_pred_noise=...)
elif self.__class__.__name__ in ['DDIM', 'LDDM']:
    # Learned model: only x and t are used
    call = self.model.predictions
    call(x, t)