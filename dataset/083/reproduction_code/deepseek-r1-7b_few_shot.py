import cross_vit

try:
    from vit_pytorch import vit_model  # Assuming this is no longer available due to migration mistake
except ImportError as e:
    print(f"Error importing vit_model: {e}")

import cross_vit

try:
    # Attempting to import a function that no longer exists under this name due to migration
    from vit_pytorch import some_non_existant_function
except ImportError as e:
    print(f"Error importing 'some_non_existant_function' from 'vit_pytorch': {e}")