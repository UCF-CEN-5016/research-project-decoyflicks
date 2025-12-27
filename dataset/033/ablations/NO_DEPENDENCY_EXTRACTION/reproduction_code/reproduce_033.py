import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import TFTBinaryDataset, sample_data
from modeling import SE3TransformerPooled
from configuration import CONFIGS
from fiber import Fiber  # Assuming Fiber is defined in a module named fiber

def load_dataset(args):
    train_split = TFTBinaryDataset(os.path.join(args.data_path, 'train.bin'), CONFIGS[args.dataset]())
    train_loader = DataLoader(train_split, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    return train_loader

def main(args):
    torch.manual_seed(42)
    args.batch_size = 64
    args.data_path = './data'  # Adjust path as necessary
    train_loader = load_dataset(args)

    # Initialize node features for modified input
    node_feats = {'0': None, '1': None}  
    for batch in train_loader:
        # Extract modified node features
        node_feats['0'] = batch['attr'][:, :5, None]
        node_feats['1'] = batch['attr'][:, 5:6, None]
        
        # Initialize the model with the correct fiber dimensions
        model = SE3TransformerPooled(
            fiber_in=Fiber({0: 5, 1: 1}),
            fiber_out=Fiber({0: args.num_degrees * args.num_channels}),  # Updated to match expected output
            fiber_edge=Fiber({0: datamodule.EDGE_FEATURE_DIM}),  # Assuming datamodule is defined elsewhere
            output_dim=1,
            tensor_cores=True
        )
        
        try:
            predictions = model(node_feats)
            # Check for the expected shape to reproduce the bug
            assert predictions.shape == (8910, 3), f"Expected shape (8910, 3), got {predictions.shape}"
        except RuntimeError as e:
            print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=CONFIGS.keys())
    parser.add_argument('--num_degrees', type=int, default=3)  # Added for completeness
    parser.add_argument('--num_channels', type=int, default=1)  # Added for completeness
    ARGS = parser.parse_args()
    main(ARGS)