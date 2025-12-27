import torch
from torchvision import transforms
from PIL import Image
from vit_pytorch import ViT

# Initialize the ViT model with specific parameters
model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000
)

# Define the transformation pipeline for input images
transform = transforms.Compose([
    transforms.Resize(224),  # Resize to expected image size
    transforms.CenterCrop(224),  # Crop the center to ensure compatibility with patch size
    transforms.ToTensor()
])

# Load and preprocess the image
image = Image.open("test_image.jpg")
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Pass the input to the model and handle any potential errors
try:
    output = model(input_tensor)
    print("No error occurred. The bug may require specific model configurations.")
except ValueError as e:
    print(f"Error occurred: {e}")
    # Handle the specific error related to shape mismatch during patch embedding