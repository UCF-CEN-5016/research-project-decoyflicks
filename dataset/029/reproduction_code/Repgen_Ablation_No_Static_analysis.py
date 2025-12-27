import argparse
import time
import torch
from torch import Adam
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import CenterCrop
from torchvision import Compose
from torchvision import ImageFolder
from torchvision import Lambda
from torchvision import Resize
from torchvision import ToTensor

from transformer_net import TransformerNet
from utils import gram_matrix, load_image, normalize_batch, save_image
from vgg import Vgg16

# Set up the argument parser
parser = argparse.ArgumentParser(description="parser for fast-neural-style")
subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
train_arg_parser.add_argument("--epochs", type=int, default=2)
train_arg_parser.add_argument("--batch-size", type=int, default=4)
train_arg_parser.add_argument("--dataset", type=str, required=True)
train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg")
train_arg_parser.add_argument("--save-model-dir", type=str, required=True)
train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None)
train_arg_parser.add_argument("--image-size", type=int, default=256)
train_arg_parser.add_argument("--style-size", type=int, default=None)
train_arg_parser.add_argument("--cuda", type=int, required=True)
train_arg_parser.add_argument("--seed", type=int, default=42)
train_arg_parser.add_argument("--content-weight", type=float, default=1e5)
train_arg_parser.add_argument("--style-weight", type=float, default=1e10)
train_arg_parser.add_argument("--lr", type=float, default=1e-3)
train_arg_parser.add_argument("--log-interval", type=int, default=500)
train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000)

eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
eval_arg_parser.add_argument("--content-image", type=str, required=True)
eval_arg_parser.add_argument("--content-scale", type=float, default=None)
eval_arg_parser.add_argument("--output-image", type=str, required=True)
eval_arg_parser.add_argument("--model", type=str, required=True)
eval_arg_parser.add_argument("--cuda", type=int, default=False)
eval_arg_parser.add_argument("--export_onnx", type=str)
eval_arg_parser.add_argument('--mps', action='store_true', default=False)

args = parser.parse_args()

# Check for CUDA availability
if args.cuda and not torch.cuda.is_available():
    print("ERROR: cuda is not available, try running on CPU")
    exit(1)

if not args.mps and torch.backends.mps.is_available():
    print("WARNING: mps is available, run with --mps to enable macOS GPU")

# Set random seed for reproducibility
torch.manual_seed(args.seed)

# Load the style image
style_image = load_image(args.style_image)
style_image = normalize_batch(style_image).unsqueeze(0)

if args.cuda:
    style_image = style_image.cuda()

# Define the transformation
transform = Compose([
    Resize(args.image_size),
    CenterCrop(args.image_size),
    ToTensor(),
])

# Load the dataset
dataset = ImageFolder(args.dataset, transform)
num_train, num_val = int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)
train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize the model and optimizer
model = TransformerNet()
optimizer = Adam(model.parameters(), args.lr)

if args.cuda:
    model.cuda()

# Training loop
for epoch in range(args.epochs):
    for i, (images, _) in enumerate(train_loader):
        if args.cuda:
            images = images.cuda()

        # Forward pass
        stylized_images = model(images)
        content_loss = torch.mean((stylized_images - images) ** 2)

        style_loss = 0.
        for f1, f2 in zip(model.content_feature_extractor(images), model.content_feature_extractor(stylized_images)):
            g1, g2 = gram_matrix(f1), gram_matrix(f2)
            style_loss += torch.mean((g1 - g2) ** 2)

        total_loss = args.content_weight * content_loss + args.style_weight * style_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (i+1) % args.log_interval == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}/{len(train_loader)}], Total Loss: {total_loss.item():.4f}")