import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

def reproduce_bug():
    # Set up parameters
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 3
    )
    
    # Create a GaussianDiffusion instance
    # Note: In the buggy version, this doesn't properly set a device property
    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000
    )
    
    # Create a batch of noise strength values
    batch_size = 4
    noise_strength = torch.ones(batch_size)
    
    print("Attempting to call offset_noise_strength...")
    try:
        # This should fail due to missing device property
        adjusted_noise = diffusion.offset_noise_strength(noise_strength, offset_noise_scale=0.1)
        print("✗ Bug not reproduced - method executed without error")
        return False
    
    except AttributeError as e:
        if "object has no attribute 'device'" in str(e):
            print(f"✓ Bug reproduced: {e}")
            return True
        else:
            print(f"Different error: {e}")
            return False
    
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    reproduce_bug()