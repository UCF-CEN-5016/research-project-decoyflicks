import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, LearnedGaussianDiffusion

def reproduce_bug():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define parameters
    image_size = 32
    channels = 3
    batch_size = 4
    
    # Create a UNet model for the diffusion process
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = channels
    ).to(device)
    
    # Create a LearnedGaussianDiffusion model
    diffusion = LearnedGaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 1000,
        objective = 'pred_v'
    ).to(device)
    
    print("Starting DDIM sampling (this should trigger the bug)...")
    
    try:
        # Try to use DDIM sampling (should raise TypeError)
        samples = diffusion.ddim_sample(
            batch_size=batch_size,
            clip_denoised=True
        )
        print("No error occurred (unexpected)")
        return False
    
    except TypeError as e:
        if "model_predictions() got multiple values for argument 'clip_x_start'" in str(e):
            print(f"✓ Bug reproduced: {e}")
            return True
        else:
            print(f"Different TypeError occurred: {e}")
            return False
    
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    reproduce_bug()