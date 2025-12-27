import argparse
from deepspeed import DeepSpeedEngine
from dschat import MiniDataset
import torch

def create_mock_data(batch_size):
    actor_loss = torch.randn(batch_size)
    critic_loss = torch.randn(batch_size)
    rewards = torch.randn(batch_size)
    return actor_loss, critic_loss, rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    batch_size = args.batch_size
    actor_loss, critic_loss, rewards = create_mock_data(batch_size)

    config = {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8
        }
    }

    ds_engine = DeepSpeedEngine(
        model=None, 
        config=config, 
        args=args, 
        train_batch_size=batch_size, 
        eval_batch_size=batch_size
    )

    dataset = MiniDataset(actor_loss, critic_loss, rewards)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for epoch in range(1):
        ds_engine.train()
        for step, (actor_input, critic_input, reward) in enumerate(dataloader):
            output = ds_engine(actor_input, critic_input)
            loss = output['loss']
            overflow = ds_engine.overflow
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}, Overflow: {overflow}")

if __name__ == "__main__":
    main()