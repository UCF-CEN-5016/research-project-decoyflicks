import torch
from torchvision import transforms as T
from dino import Dino, NetWrapper

# Set seed for reproducibility
torch.manual_seed(42)

# Define batch size and image dimensions
batch_size = 32
height = width = 256

# Load the Dino model with specified parameters
dino_model = Dino(
    net=None,
    image_size=height,
    hidden_layer=-2,
    projection_hidden_size=256,
    num_classes_K=65336,
    projection_layers=4,
    student_temp=0.9,
    teacher_temp=0.04,
    local_upper_crop_scale=0.4,
    global_lower_crop_scale=0.5,
    moving_average_decay=0.9,
    center_moving_average_decay=0.9
)

# Define augmentation functions if not provided (default to None)
augment_fn = None
augment_fn2 = None

# Create random uniform input data with shape (batch_size, 3, height, width)
input_data = torch.rand(batch_size, 3, height, width)

# Call the forward method of the Dino class
student_proj_one, student_proj_two = dino_model(input_data, return_embedding=False, return_projection=True)

# Print output tensors to verify NaN values
print("Student Proj One:", student_proj_one)
print("Student Proj Two:", student_proj_two)