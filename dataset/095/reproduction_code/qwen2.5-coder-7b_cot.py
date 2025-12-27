from typing import Any, Callable, Dict, Tuple


class SamplerBase:
    """
    Base sampler that routes prediction calls to the appropriate model
    depending on the concrete sampler class name.

    Core behavior:
    - For class named 'DDPM', call gaussian_model.predictions(x, t, clip_x_start=..., rederive_pred_noise=...)
    - For class named 'DDIM' or 'LDDM', call model.predictions(x, t)
    """

    def _prediction_callable(
        self, clip_x_start: bool, rederive_pred_noise: bool
    ) -> Tuple[Callable[..., Any], Dict[str, Any]]:
        """
        Return a (callable, kwargs) tuple appropriate for this sampler subclass.

        Do not change logic: branch solely on the class name.
        """
        cls_name = self.__class__.__name__
        if cls_name == "DDPM":
            return (
                self.gaussian_model.predictions,
                {"clip_x_start": clip_x_start, "rederive_pred_noise": rederive_pred_noise},
            )
        if cls_name in ("DDIM", "LDDM"):
            return self.model.predictions, {}
        raise NotImplementedError(f"Prediction routing not implemented for sampler '{cls_name}'")

    def model_predictions(
        self, x: Any, t: Any, clip_x_start: bool = True, rederive_pred_noise: bool = True
    ) -> Any:
        """
        Wrapper that calls the chosen prediction callable with the correct arguments.
        """
        pred_callable, extra_kwargs = self._prediction_callable(clip_x_start, rederive_pred_noise)
        return pred_callable(x, t, **extra_kwargs)

    def ddim_sample(
        self, sample_x: Any, timestep: Any, clip_x_start: bool = True, rederive_pred_noise: bool = True
    ) -> Any:
        """
        Perform a DDIM sampling step by delegating to model_predictions.

        Keeps original branching behavior intact via model_predictions.
        """
        return self.model_predictions(sample_x, timestep, clip_x_start=clip_x_start, rederive_pred_noise=rederive_pred_noise)