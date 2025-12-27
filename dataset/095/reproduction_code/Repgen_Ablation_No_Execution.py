import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, LearnedGaussianDiffusion
import inspect
import traceback

def reproduce_bug_detailed():
    print("=== Step 1: Setting up environment ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define model parameters
    image_size = 32
    channels = 3
    timesteps = 100  # Using fewer timesteps for faster execution
    
    print("\n=== Step 2: Creating UNet model ===")
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels = channels
    ).to(device)
    print(f"UNet model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    print("\n=== Step 3: Creating LearnedGaussianDiffusion model ===")
    diffusion = LearnedGaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = timesteps,
        objective = 'pred_v'
    ).to(device)
    print("LearnedGaussianDiffusion model created")
    
    print("\n=== Step 4: Examining method signatures ===")
    # Check signatures of the model_predictions methods
    print("GaussianDiffusion.model_predictions signature:")
    try:
        base_sig = inspect.signature(GaussianDiffusion.model_predictions)
        print(f"  {base_sig}")
    except Exception as e:
        print(f"  Error getting signature: {e}")
    
    print("\nLearnedGaussianDiffusion.model_predictions signature:")
    try:
        learned_sig = inspect.signature(LearnedGaussianDiffusion.model_predictions)
        print(f"  {learned_sig}")
    except Exception as e:
        print(f"  Error getting signature: {e}")
    
    # Check signature of ddim_sample method
    print("\nGaussianDiffusion.ddim_sample signature:")
    try:
        ddim_sig = inspect.signature(GaussianDiffusion.ddim_sample)
        print(f"  {ddim_sig}")
    except Exception as e:
        print(f"  Error getting signature: {e}")
    
    print("\n=== Step 5: Testing DDIM sampling ===")
    batch_size = 2
    
    # Try to use DDIM sampling
    try:
        print("Attempting DDIM sampling...")
        samples = diffusion.ddim_sample(
            batch_size=batch_size,
            clip_denoised=True
        )
        print("✗ Sampling completed without errors (unexpected)")
        
    except TypeError as e:
        print(f"✓ TypeError encountered as expected: {e}")
        print("\nStacktrace:")
        traceback.print_exc()
        
        print("\nAnalyzing the error:")
        if "model_predictions() got multiple values for argument 'clip_x_start'" in str(e):