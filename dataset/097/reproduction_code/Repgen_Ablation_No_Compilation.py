import torch
import inspect
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

def reproduce_bug_detailed():
    print("=== Step 1: Examining GaussianDiffusion implementation ===")
    
    # Get the source code for the __init__ method if possible
    try:
        init_source = inspect.getsource(GaussianDiffusion.__init__)
        print("GaussianDiffusion.__init__ method:")
        print(init_source[:200] + "..." if len(init_source) > 200 else init_source)
        
        # Check if 'device' is stored as an attribute
        if "self.device" in init_source:
            print("'self.device' is set in __init__")
        else:
            print("'self.device' is NOT set in __init__ (potential issue)")
    except Exception as e:
        print(f"Could not inspect source: {e}")
    
    # Look for device property or getter method
    if hasattr(GaussianDiffusion, 'device'):
        if isinstance(GaussianDiffusion.device, property):
            print("GaussianDiffusion has a 'device' property")
        else:
            print("GaussianDiffusion has a 'device' attribute (not a property)")
    else:
        print("GaussianDiffusion does NOT have a 'device' property or attribute")
    
    # Look for offset_noise_strength method
    if hasattr(GaussianDiffusion, 'offset_noise_strength'):
        try:
            method_source = inspect.getsource(GaussianDiffusion.offset_noise_strength)
            print("\nGaussianDiffusion.offset_noise_strength method:")
            print(method_source)
            
            if "self.device" in method_source:
                print("Method uses 'self.device' (will fail if not defined)")
        except Exception as e:
            print(f"Could not inspect method: {e}")
    else:
        print("\nGaussianDiffusion does NOT have an 'offset_noise_strength' method")
        # This could mean we're using an older version or the method has a different name
    
    print("\n=== Step 2: Creating model and diffusion instance ===")
    model = Unet(
        dim = 64,
        dim_mults = (1, 2),  # Smaller for faster initialization
        channels = 3
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size = 64,  # Smaller for faster initialization
        timesteps = 100   # Fewer timesteps for faster initialization
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Check if diffusion instance has device attribute
    if hasattr(diffusion, 'device'):
        print(f"diffusion.device = {diffusion.device}")
    else:
        print("diffusion instance does NOT have a 'device' attribute")
    
    print("\n=== Step 3: Attempting to call offset_noise_strength ===")
    batch_size = 4
    noise_strength = torch.ones(batch_size)
    
    try:
        # This should fail if there's no device property/attribute
        adjusted_noise = diffusion.offset_noise_strength(noise_strength, offset_noise_scale=0.1)
        print("✗ Bug not reproduced - method executed without error")
        print(f"Adjusted noise shape: {adjusted_noise.shape}")
    except AttributeError as e:
        if "object has no attribute 'device'" in str(e):
            print(f"✓ Bug reproduced: {e}")
            print("\nThe issue is confirmed: GaussianDiffusion is missing a 'device' property")
            print("but the offset_noise_strength method tries to use it.")
        else:
            print(f"Different error: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
    
    print("\n=== Step 4: Examining possible fixes ===")
    print("A proper fix would be to add a device property to GaussianDiffusion:")
    print("@property")
    print("def device(self):")
    print("    return self.betas.device  # Or another tensor's device")
    
    # Try a quick monkey patch to fix the issue
    print("\nTesting a monkey patch fix...")
    
    # Add a device property to the instance
    try:
        # Find a tensor attribute to get its device
        for attr_name in dir(diffusion):
            attr = getattr(diffusion, attr_name)
            if isinstance(attr, torch.Tensor):
                diffusion.device = attr.device
                print(f"Added device property based on {attr_name}: {diffusion.device}")
                break
        
        # Try again with the fix
        adjusted_noise = diffusion.offset_noise_strength(noise_strength, offset_noise_scale=0.1)
        print("✓ Fix successful - method executed without error")
        print(f"Adjusted noise shape: {adjusted_noise.shape}")
    except Exception as e:
        print(f"Fix failed: {e}")

if __name__ == "__main__":
    reproduce_bug_detailed()