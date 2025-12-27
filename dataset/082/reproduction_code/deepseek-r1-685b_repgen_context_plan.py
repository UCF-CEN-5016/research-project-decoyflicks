import torch
from torch.optim import Adam
from vit_pytorch import ViT
from vit_pytorch.mpp import MPP

class ModelConfig:
    """
    Configuration class for ViT and MPP model parameters.
    """
    def __init__(self):
        self.image_size = 256
        self.patch_size = 32
        self.num_classes = 1000
        self.dim = 1024
        self.depth = 6
        self.heads = 8
        self.mlp_dim = 2048
        self.dropout = 0.1
        self.emb_dropout = 0.1
        self.mask_prob = 0.15
        self.random_patch_prob = 0.30
        self.replace_prob = 0.50
        self.learning_rate = 3e-4
        self.num_training_steps = 100
        self.batch_size = 20
        self.image_channels = 3
        self.output_path = './pretrained-net.pt'

def initialize_vit_model(config: ModelConfig) -> ViT:
    """
    Initializes the Vision Transformer (ViT) model.
    """
    model = ViT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout
    )
    print("ViT model initialized.")
    return model

def initialize_mpp_trainer(vit_model: ViT, config: ModelConfig) -> MPP:
    """
    Creates an MPP trainer using the ViT model.
    """
    mpp_trainer = MPP(
        transformer=vit_model,
        patch_size=config.patch_size,
        dim=config.dim,
        mask_prob=config.mask_prob,
        random_patch_prob=config.random_patch_prob,
        replace_prob=config.replace_prob
    )
    print("MPP trainer initialized.")
    return mpp_trainer

def setup_optimizer(mpp_trainer: MPP, config: ModelConfig) -> Adam:
    """
    Sets up the Adam optimizer for the MPP trainer.
    """
    opt = Adam(mpp_trainer.parameters(), lr=config.learning_rate)
    print(f"Optimizer (Adam with LR={config.learning_rate}) set up.")
    return opt

def sample_unlabelled_images(config: ModelConfig) -> torch.Tensor:
    """
    Generates a batch of dummy unlabelled images.
    """
    return torch.FloatTensor(
        config.batch_size,
        config.image_channels,
        config.image_size,
        config.image_size
    ).uniform_(0., 1.)

def train_mpp_model(mpp_trainer: MPP, optimizer: Adam, config: ModelConfig):
    """
    Runs the training loop for the MPP model.
    """
    print(f"Starting MPP training for {config.num_training_steps} steps...")
    mpp_trainer.train() # Set trainer to training mode
    for i in range(config.num_training_steps):
        images = sample_unlabelled_images(config)
        
        # If CUDA is available, move images and model to GPU
        if torch.cuda.is_available():
            images = images.cuda()
            mpp_trainer = mpp_trainer.cuda()

        loss = mpp_trainer(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/{config.num_training_steps}, Loss: {loss.item():.4f}")
    print("MPP training complete.")

def save_model_weights(model: ViT, output_path: str):
    """
    Saves the state dictionary of the ViT model.
    """
    torch.save(model.state_dict(), output_path)
    print(f"Model state dictionary saved to: {output_path}")

def main():
    """
    Main function to orchestrate the ViT MPP training process.
    """
    config = ModelConfig()

    vit_model = initialize_vit_model(config)
    mpp_trainer = initialize_mpp_trainer(vit_model, config)
    optimizer = setup_optimizer(mpp_trainer, config)
    
    train_mpp_model(mpp_trainer, optimizer, config)
    save_model_weights(vit_model, config.output_path)

if __name__ == "__main__":
    main()
