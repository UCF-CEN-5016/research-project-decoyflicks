import torch
from denoising_diffusion_pytorch import Unet, LearnedGaussianDiffusion

def minimal_reproduction():
    # Create a minimal setup to reproduce the bug
    image_size = 16  # Very small image for quick testing
    channels = 1     # Just one channel
    
    # Create a small UNet model
    model = Unet(
        dim = 32,
        dim_mults = (1, 2),
        channels = channels
    )
    
    # Create LearnedGaussianDiffusion
    diffusion = LearnedGaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 10,  # Very few timesteps for quick testing
        objective = 'pred_v'
    )
    
    # Try to use DDIM sampling - this should trigger the TypeError
    try:
        samples = diffusion.ddim_sample(
            batch_size=1,
            clip_denoised=True
        )
        print("No error occurred - bug not reproduced")
        
    except TypeError as e:
        if "model_predictions() got multiple values for argument 'clip_x_start'" in str(e):
            print("✓ Bug reproduced successfully")
            print(f"Error: {e}")
            
            # Show the issue
            print("\nThe issue occurs because:")
            print("1. GaussianDiffusion.ddim_sample() calls:")
            print("   self.model_predictions(x, t, clip_x_start=clip_denoised)")
            print("2. But LearnedGaussianDiffusion.model_predictions() has signature:")
            print("   def model_predictions(self, x, t, clip_x_start=True, rederive_pred_noise=False)")
            print("3. This causes 'clip_x_start' to be passed twice")
            
            return True
        else:
            print(f"Different TypeError occurred: {e}")
            return False
            
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return False
        
    return False

if __name__ == "__main__":
    minimal_reproduction()