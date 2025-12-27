import argparse
import os
import logging
import subprocess
import time
import requests
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# API configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

class CodeGenerator:
    """Handles code generation using advanced prompting techniques"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.examples = self._get_high_quality_examples()
    
    def generate(self, bug_report: str, technique: str = "zero_shot", n_examples = 3) -> str:
        """Generate reproduction code using specified technique"""
        prompt = self._build_prompt(bug_report, technique, n_examples)
        
        if self.model_name.startswith("llama3-8b") or self.model_name.startswith("qwen3-8b") or self.model_name.startswith("deepseek-r1-7b") or self.model_name.startswith("qwen2.5-7b"):
            logger.info(f"Using local Ollama model: {self.model_name}")
            return self._ollama_local(prompt)
        elif self.model_name.startswith("deepseek"):
            return self._deepseek_api(prompt)
        elif self.model_name.startswith("llama3.3-70b"):
            return self._groq_api(prompt)
        elif self.model_name.startswith("gpt-4.1"):
            return self._openai_api(prompt)
    
    def _build_prompt(self, bug_report: str, technique: str, n_examples = 3) -> str:
        """Build optimized prompt based on technique"""
        if technique == "zero_shot":
            return self._zero_shot_prompt(bug_report)
        elif technique == "few_shot":
            return self._few_shot_prompt(bug_report, n_examples)
        elif technique == "cot":
            return self._cot_prompt(bug_report)
        else:
            raise ValueError(f"Unknown technique: {technique}")
    
    def _zero_shot_prompt(self, bug_report: str) -> str:
        """Basic zero-shot prompt"""
        return f"""Generate a minimal, self-contained Python script to reproduce this bug:

Bug Report:
{bug_report}

Requirements:
1. Complete runnable code with all necessary imports
2. Minimal setup to reproduce the issue
3. No explanations or comments
4. Code wrapped in ```python``` block"""

    def _few_shot_prompt(self, bug_report: str, n_examples = 3) -> str:
        """Few-shot prompt with carefully selected examples"""
        examples = "\n\n".join(
            f"===== EXAMPLE {i+1} =====\n"
            f"Bug Type: {ex['type']}\n"
            f"Bug Description: {ex['description']}\n"
            f"Reproduction Code:\n{ex['code']}"
            for i, ex in enumerate(self.examples[:n_examples])
        )
        
        return f"""Below are examples of high-quality bug reproduction code. Each example shows:
1. The bug type
2. A clear description
3. Minimal reproduction code

{examples}

Now generate reproduction code for this new bug:

Bug Report:
{bug_report}

Guidelines:
1. Follow the same format as the examples
2. Keep the code minimal but complete
3. Include all necessary imports
4. Wrap the final code in ```python```"""

    def _cot_prompt(self, bug_report: str) -> str:
        """Chain-of-Thought prompt"""
        return f"""Let's analyze and reproduce this bug step by step:

Bug Report:
{bug_report}

Thinking Process:
1. What are the key symptoms of this bug?
2. What Python components are needed to reproduce it?
3. What minimal setup is required?
4. What specific conditions trigger the bug?
5. How can we isolate the core issue?

Now generate the reproduction code:
1. Start with necessary imports
2. Set up minimal environment
3. Add triggering conditions
4. Wrap final code in ```python```"""

    def _get_high_quality_examples(self) -> list:
        """Return carefully curated examples for few-shot learning"""
        return [
        {
            "type": "Gradient Vanishing",
            "description": "Gradients become too small in deep networks with sigmoid activation",
            "context": "Training deep neural network for image classification",
            "components": ["torch.nn", "sigmoid activation", "backpropagation"],
            "expected": "Gradients flowing through all layers",
            "actual": "Near-zero gradients in early layers",
            "code": """
import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.Sigmoid(),  # Problem: Using sigmoid in deep network
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

model = DeepNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# After training
for name, param in model.named_parameters():
    if 'weight' in name:
        print(f"{name} gradient norm: {param.grad.norm()}")  # Early layers show tiny gradients
"""
        },
        {
            "type": "Shape Mismatch in CNN",
            "description": "Incorrect tensor shapes in CNN architecture",
            "context": "CNN model with inconsistent layer dimensions",
            "components": ["torch.nn.Conv2d", "torch.nn.Linear", "flatten operation"],
            "expected": "Compatible tensor dimensions throughout network",
            "actual": "RuntimeError: shape mismatch in fully connected layer",
            "code": """
import torch
import torch.nn as nn

class BadCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)  # Output: [batch, 16, 30, 30]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # Output: [batch, 32, 28, 28]
        self.fc1 = nn.Linear(32*28*28, 10)  # Problem: Incorrect dimension calculation
    
    def forward(self, x):  # Input: [batch, 3, 32, 32]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc1(x)

model = BadCNN()
x = torch.randn(4, 3, 32, 32)
output = model(x)  # Raises RuntimeError due to shape mismatch
"""
        },
        {
            "type": "Loss Explosion with Normalization",
            "description": "Unstable training due to incorrect batch normalization usage",
            "context": "Training with batch normalization in eval mode",
            "components": ["torch.nn.BatchNorm2d", "model.eval()", "training loop"],
            "expected": "Stable training with normalized activations",
            "actual": "Exploding loss values",
            "code": """
import torch
import torch.nn as nn

class UnstableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64*30*30, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  # Problem: BatchNorm in eval mode with single sample
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = UnstableNet()
model.eval()  # Setting to eval mode incorrectly
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for i in range(100):  # Single sample batches
        x = torch.randn(1, 3, 32, 32)  # Problem: Single sample with BatchNorm
        optimizer.zero_grad()
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")  # Will show exploding values
"""
        },
        {
            "type": "CUDA Memory Leak",
            "description": "Memory accumulation due to retained computational graph",
            "context": "Training loop without gradient clearing",
            "components": ["torch.cuda", "backward pass", "memory management"],
            "expected": "Constant GPU memory usage",
            "actual": "Gradually increasing GPU memory until OOM",
            "code": """
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000)
).cuda()

optimizer = torch.optim.Adam(model.parameters())
losses = []  # Problem: Storing computation graph

for i in range(1000):
    x = torch.randn(100, 1000).cuda()
    y = torch.randn(100, 1000).cuda()
    
    output = model(x)
    loss = nn.MSELoss()(output, y)
    losses.append(loss)  # Problem: Accumulating computational graphs
    
    if i % 100 == 0:
        torch.cat(losses).sum().backward()  # Memory leak: Huge computational graph
        optimizer.step()
        
    print(f"GPU Memory: {torch.cuda.memory_allocated()}")  # Shows increasing memory
"""
        },
        {
            "type": "NaN Loss with Large Learning Rate",
            "description": "Training instability with large learning rate in transformer",
            "context": "Training transformer model with improper learning rate warmup",
            "components": ["torch.nn.TransformerEncoder", "learning rate scheduler"],
            "expected": "Stable training with gradual learning rate increase",
            "actual": "NaN loss due to unstable gradients",
            "code": """
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
    def forward(self, x):
        return self.transformer(x)

model = SimpleTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Problem: Too large learning rate

# Training loop without warmup
for epoch in range(10):
    x = torch.randn(20, 32, 512)  # [seq_len, batch, features]
    y = torch.randn(20, 32, 512)
    
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    # Problem: No gradient clipping
    optimizer.step()
    print(f"Loss: {loss.item()}")  # Will show NaN after few iterations
"""
        }, {
            "type": "DataLoader Deadlock",
            "description": "Multiprocessing deadlock in custom dataset",
            "context": "DataLoader with custom collate function",
            "components": ["torch.utils.data", "DataLoader", "num_workers"],
            "expected": "Efficient parallel data loading",
            "actual": "Process hanging due to deadlock",
            "code": """
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class ProblematicDataset(Dataset):
    def __init__(self):
        self.data = [np.random.randn(100) for _ in range(1000)]
    
    def __getitem__(self, idx):
        # Problem: Non-picklable objects in __getitem__
        self.temporary_cache = {}  # Instance attribute modification
        return torch.from_numpy(self.data[idx])
    
    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    return torch.stack(batch)

dataset = ProblematicDataset()
loader = DataLoader(dataset, 
                   batch_size=32, 
                   num_workers=4,  # Will cause deadlock
                   collate_fn=collate_fn)

for batch in loader:  # Hangs indefinitely
    print(batch.shape)
"""
        },
        {
            "type": "Multi-GPU Parameter Mismatch",
            "description": "Inconsistent parameter updates across GPUs",
            "context": "DistributedDataParallel training",
            "components": ["torch.nn.parallel", "DistributedDataParallel"],
            "expected": "Synchronized parameter updates",
            "actual": "Diverging model parameters across GPUs",
            "code": """
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        # Problem: Using buffer without registration
        self.register_buffer('running_mean', torch.zeros(10))
        self.unregistered_buffer = torch.zeros(10)  # Will cause sync issues

    def forward(self, x):
        self.unregistered_buffer += 0.1  # Updates won't sync across GPUs
        return self.fc(x + self.unregistered_buffer)

def train(rank, world_size):
    setup(rank, world_size)
    model = Net().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Training loop will have inconsistent behavior
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(100):
        optimizer.step()
        print(f"Rank {rank} buffer:", model.module.unregistered_buffer)
"""
        },
        {
            "type": "Custom Loss Function Gradient Error",
            "description": "Incorrect gradient computation in custom loss",
            "context": "Implementation of novel loss function",
            "components": ["torch.autograd", "custom loss", "backward hook"],
            "expected": "Proper gradient flow",
            "actual": "Incorrect gradients due to in-place operations",
            "code": """
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        # Problem: In-place operations breaking computational graph
        diff = pred - target
        diff.pow_(2)  # In-place operation
        loss = diff.mean()
        
        # Another problem: Modifying tensors in-place
        pred.add_(1.0)  # Modifies input tensor
        
        return loss

model = nn.Linear(10, 10)
criterion = CustomLoss()
optimizer = torch.optim.Adam(model.parameters())

x = torch.randn(32, 10)
target = torch.randn(32, 10)

for _ in range(100):
    pred = model(x)
    loss = criterion(pred, target)  # Will produce incorrect gradients
    loss.backward()
    optimizer.step()
"""
        },
        {
            "type": "Attention Mask Broadcasting Error",
            "description": "Incorrect attention mask shape in transformer",
            "context": "Custom attention mechanism implementation",
            "components": ["torch.nn.MultiheadAttention", "mask broadcasting"],
            "expected": "Correct attention masking",
            "actual": "Broadcasting error or incorrect attention patterns",
            "code": """
class CustomAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Problem: Incorrect mask shape and broadcasting
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        if mask is not None:
            # mask shape: [batch_size, seq_len]
            # scores shape: [batch_size, seq_len, seq_len]
            scores = scores.masked_fill(mask, float('-inf'))  # Will raise error
            
        attention = torch.softmax(scores, dim=-1)
        return torch.matmul(attention, V)

# Usage
batch_size, seq_len, dim = 32, 50, 64
x = torch.randn(batch_size, seq_len, dim)
mask = torch.zeros(batch_size, seq_len).bool()  # Incorrect mask shape
model = CustomAttention(dim)
output = model(x, mask)  # Raises broadcasting error
"""
        },
        {
            "type": "Transfer Learning Layer Mismatch",
            "description": "Incompatible layer dimensions in transfer learning",
            "context": "Fine-tuning pretrained model",
            "components": ["torchvision.models", "model adaptation"],
            "expected": "Successful model adaptation",
            "actual": "Dimension mismatch in modified layers",
            "code": """
import torchvision.models as models

def modify_resnet(num_classes=10):
    model = models.resnet50(pretrained=True)
    
    # Problem: Incorrect handling of final layer dimensions
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    
    # Problem: Modifying intermediate layers incorrectly
    model.layer4[0].conv1 = nn.Conv2d(
        1024, 512, kernel_size=1, stride=2  # Wrong input channels
    )
    
    return model

model = modify_resnet()
x = torch.randn(16, 3, 224, 224)
output = model(x)  # Raises dimension mismatch error
"""
        },
        {
            "type": "Regularization Implementation Bug",
            "description": "Incorrect application of regularization",
            "context": "Custom regularization in training loop",
            "components": ["L1/L2 regularization", "gradient modification"],
            "expected": "Proper regularization effect",
            "actual": "Over-regularization or no effect",
            "code": """
class CustomRegularization:
    def __init__(self, model, l1_factor=0.01, l2_factor=0.01):
        self.model = model
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        
    def compute_penalty(self):
        l1_reg = 0
        l2_reg = 0
        
        for param in self.model.parameters():
            # Problem: Adding regularization to all parameters including bias
            l1_reg += torch.sum(torch.abs(param))
            l2_reg += torch.sum(param ** 2)
            
        # Problem: Applying regularization twice
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data.add_(self.l1_factor * torch.sign(param.data))
                param.grad.data.add_(self.l2_factor * 2 * param.data)
                
        return self.l1_factor * l1_reg + self.l2_factor * l2_reg

model = nn.Linear(10, 10)
regularizer = CustomRegularization(model)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)  # Double regularization

for _ in range(100):
    # Training loop with incorrect regularization
    loss = criterion(model(x), y) + regularizer.compute_penalty()
    loss.backward()
    optimizer.step()
"""
        },
        {
            "type": "LSTM State Management Error",
            "description": "Incorrect handling of hidden states in LSTM",
            "context": "Sequential data processing with LSTM",
            "components": ["torch.nn.LSTM", "hidden state management"],
            "expected": "Proper sequence processing",
            "actual": "Memory leak or incorrect state propagation",
            "code": """
class StatefulLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden = None
        
    def forward(self, x):
        # Problem: Hidden state not detached between batches
        if self.hidden is None:
            output, self.hidden = self.lstm(x)
        else:
            output, self.hidden = self.lstm(x, self.hidden)
            
        return output
    
    def reset_hidden(self, batch_size):
        # Problem: Hidden state on wrong device
        self.hidden = (torch.zeros(1, batch_size, self.lstm.hidden_size),
                      torch.zeros(1, batch_size, self.lstm.hidden_size))

model = StatefulLSTM(input_size=10, hidden_size=20).cuda()
for batch in dataloader:
    x = batch.cuda()
    model.reset_hidden(x.size(0))  # Hidden state remains on CPU
    output = model(x)  # Device mismatch error
"""
        },
        {
            "type": "Embedding Layer Padding Issue",
            "description": "Incorrect handling of padding indices in embeddings",
            "context": "Text processing with embeddings",
            "components": ["torch.nn.Embedding", "padding_idx"],
            "expected": "Proper handling of padding tokens",
            "actual": "Gradient updates to padding embeddings",
            "code": """
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Problem: Padding index not specified
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, x, lengths):
        # Problem: Not masking padding values
        embedded = self.embedding(x)
        mask = (x != 0).float().unsqueeze(-1)
        return embedded * mask  # Incorrect masking approach

# Usage with varying sequence lengths
vocab_size, embed_dim = 1000, 300
model = TextEncoder(vocab_size, embed_dim)
sequences = torch.randint(0, vocab_size, (32, 50))  # Some entries should be padding
lengths = torch.randint(1, 50, (32,))

# Padding tokens still receive gradient updates
output = model(sequences, lengths)
loss = output.sum()
loss.backward()  # Padding embeddings incorrectly updated
"""
        },
        {
            "type": "Checkpoint Loading State Mismatch",
            "description": "Incomplete state restoration from checkpoint",
            "context": "Model checkpointing and restoration",
            "components": ["state_dict", "optimizer state", "learning rate scheduler"],
            "expected": "Complete model restoration",
            "actual": "Partial state restoration leading to training issues",
            "code": """
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.running_average = None  # Custom buffer
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.running_average is None:
            self.running_average = x.mean().detach()
        return x

model = ComplexModel()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

# Problem: Incomplete state saving
checkpoint = {
    'model_state': model.state_dict(),
    # Missing optimizer and scheduler states
    # Missing custom buffer
}

torch.save(checkpoint, 'model.pth')

# Later, loading the checkpoint
new_model = ComplexModel()
new_model.load_state_dict(checkpoint['model_state'])
# Problem: Optimizer and scheduler states not restored
new_optimizer = torch.optim.Adam(new_model.parameters())  # Fresh optimizer state
"""
        },
        {
            "type": "Distributed Training Race Condition",
            "description": "Race condition in distributed metric computation",
            "context": "Multi-GPU training with metric aggregation",
            "components": ["torch.distributed", "metric computation"],
            "expected": "Correct metric aggregation",
            "actual": "Inconsistent metrics across processes",
            "code": """
class DistributedMetrics:
    def __init__(self):
        self.correct = torch.tensor(0).cuda()
        self.total = torch.tensor(0).cuda()
        
    def update(self, pred, target):
        # Problem: Race condition in metric update
        correct = (pred == target).sum()
        self.correct += correct
        self.total += len(target)
        
    def compute(self):
        # Problem: No proper synchronization
        accuracy = self.correct / self.total
        # Should use all_reduce to aggregate across GPUs
        return accuracy.item()

def train_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = DistributedDataParallel(Model().to(rank), device_ids=[rank])
    metrics = DistributedMetrics()
    
    for data, target in dataloader:
        output = model(data)
        metrics.update(output.argmax(1), target)
        
        if rank == 0:
            # Problem: Metrics computed before other processes finish
            print(f"Accuracy: {metrics.compute()}")  # Incorrect partial results
"""
        },
        {
            "type": "Custom Dataset Sampling Bug",
            "description": "Biased sampling in custom dataset",
            "context": "Implementation of custom sampling strategy",
            "components": ["torch.utils.data", "WeightedRandomSampler"],
            "expected": "Balanced class sampling",
            "actual": "Biased sampling distribution",
            "code": """
class ImbalancedDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = []
        self.labels = []
        # Highly imbalanced dataset
        self.labels = torch.randint(0, 10, (num_samples,))
        self.labels[:800] = 0  # 80% class 0
        
    def __getitem__(self, idx):
        return torch.randn(10), self.labels[idx]
        
    def __len__(self):
        return len(self.labels)

def create_balanced_sampler(dataset):
    labels = dataset.labels
    class_counts = torch.bincount(labels)
    
    # Problem: Incorrect weight calculation
    weights = 1. / class_counts[labels]  # Missing normalization
    
    # Problem: Sampler length mismatch
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset) // 2,  # Wrong number of samples
        replacement=False  # Will raise error when exhausted
    )
    return sampler

dataset = ImbalancedDataset()
sampler = create_balanced_sampler(dataset)
loader = DataLoader(dataset, sampler=sampler, batch_size=32)

# Training with incorrect sampling
for batch, labels in loader:
    # Will not see all samples and have incorrect class distribution
    pass
"""
        },
        {
            "type": "Model Architecture Dimension Bug",
            "description": "Subtle dimension mismatches in complex architecture",
            "context": "Custom neural network architecture",
            "components": ["skip connections", "feature concatenation"],
            "expected": "Compatible feature dimensions",
            "actual": "Runtime shape mismatch errors",
            "code": """
class ComplexArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Problem: Incorrect dimension calculation for skip connection
        self.upsample = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # Should be 256 (128 + 128 from skip connection)
        self.conv4 = nn.Conv2d(128, 64, 3, padding=1)
        
    def forward(self, x):
        # Feature maps for skip connections
        f1 = self.conv1(x)
        f2 = self.conv2(F.max_pool2d(f1, 2))
        f3 = self.conv3(F.max_pool2d(f2, 2))
        
        # Problem: Dimension mismatch in skip connection
        up = self.upsample(f3)
        # Should concatenate: torch.cat([up, f2], dim=1)
        out = self.conv4(up + f2)  # Wrong feature combination
        
        return out

model = ComplexArchitecture()
x = torch.randn(1, 3, 64, 64)
output = model(x)  # Raises dimension mismatch error
"""
        },
        {
            "type": "Autograd Graph Memory Leak",
            "description": "Memory leak in custom autograd function",
            "context": "Implementation of custom backward pass",
            "components": ["torch.autograd.Function", "custom gradients"],
            "expected": "Proper memory cleanup",
            "actual": "Memory accumulation over iterations",
            "code": """
class LeakyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Problem: Storing entire input tensor
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Problem: Creating new tensors without deletion
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        
        # Memory leak: Storing intermediate results
        ctx.intermediate_results = grad_input  # Unnecessary storage
        return grad_input

class LeakyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1000, 1000)
        self.leaky = LeakyFunction.apply
        self.history = []  # Problem: Storing all intermediate activations
        
    def forward(self, x):
        x = self.linear(x)
        x = self.leaky(x)
        self.history.append(x.detach())  # Memory leak
        return x

model = LeakyModel()
optimizer = torch.optim.Adam(model.parameters())

for _ in range(1000):
    x = torch.randn(32, 1000)
    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    # Memory usage grows with each iteration
"""
        },
        {
            "type": "Mixed Precision Training Error",
            "description": "Incorrect handling of FP16/FP32 conversion",
            "context": "Mixed precision training implementation",
            "components": ["torch.cuda.amp", "GradScaler"],
            "expected": "Stable mixed precision training",
            "actual": "Gradient overflow or underflow",
            "code": """
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model):
        self.model = model
        self.scaler = GradScaler()
        # Problem: Incorrect loss scaling initialization
        self.scaler._scale = torch.tensor(2**15)  # Manual scale setting
        
    def training_step(self, batch):
        x, y = batch
        
        # Problem: Inconsistent precision handling
        with autocast():
            output = self.model(x)
            # Loss computation in wrong precision
            loss = F.mse_loss(output.float(), y)
        
        # Problem: Incorrect gradient scaling
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        
        # Problem: Missing gradient unscaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Problem: Mixing FP16 and FP32 in computation
        with autocast():
            metrics = torch.tensor([1.0], device='cuda')
            metrics = metrics.half() + loss.float()  # Precision mismatch

model = ComplexModel().cuda()
trainer = MixedPrecisionTrainer(model)

for batch in dataloader:
    trainer.training_step(batch)  # Will cause gradient overflow
"""
        },
        {
            "type": "Custom Layer Initialization Bug",
            "description": "Improper weight initialization in custom layer",
            "context": "Implementation of specialized neural network layer",
            "components": ["parameter initialization", "custom nn.Module"],
            "expected": "Properly initialized weights",
            "actual": "Vanishing/exploding gradients due to poor initialization",
            "code": """
class CustomLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Problem: Poor initialization strategy
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Problem: No scaling of initial values
        nn.init.uniform_(self.weight, -1, 1)  # Too wide distribution
        
    def reset_parameters(self):
        # Problem: Incorrect fan_in calculation
        fan_in = self.weight.size(0)  # Wrong dimension
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)
        
    def forward(self, x):
        # Problem: Numerically unstable implementation
        return torch.mm(x, self.weight.t()) + self.bias

class DeepCustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomLayer(100, 100) for _ in range(10)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            # Problem: No normalization between layers
            x = torch.tanh(layer(x))
        return x

model = DeepCustomNetwork()
x = torch.randn(32, 100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for _ in range(100):
    output = model(x)
    loss = output.sum()
    loss.backward()
    # Gradients will vanish or explode
    print(f"Gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)}")
"""
        }
    ]
    
    def _ollama_local(self, prompt: str) -> str:
        """Generate code using local Ollama models"""
        model_map = {
            "llama3-8b": "llama3:8b" ,
            "qwen3-8b": "qwen3:8b",
            "deepseek-r1-7b": "deepseek-r1:7b",
            "qwen2.5-7b": "qwen2.5:7b",
            "qwen2.5-coder": "qwen2.5-coder:7b"
        }
        
        cmd = ["ollama", "run", model_map[self.model_name], prompt]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=True
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.error(f"{self.model_name} timed out after 5 minutes")
            return ""
        except subprocess.CalledProcessError as e:
            logger.error(f"{self.model_name} failed: {e.stderr}")
            return ""
    
    def _openai_api(self, prompt: str) -> str:
        """Generate code using OpenAI's GPT-4.1 API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer <OPENAI_API_KEY>",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4.1-2025-04-14",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._extract_code(content)
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return ""
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code block from model response"""
        # Look for ```python or ``` markers
        start_marker = "```python"
        end_marker = "```"
        
        start_idx = text.find(start_marker)
        if start_idx == -1:
            # Try with just ```
            start_idx = text.find(end_marker)
            if start_idx != -1:
                start_idx += len(end_marker)
        else:
            start_idx += len(start_marker)
        
        if start_idx == -1:
            # No markers found, return whole text
            return text.strip()
        
        end_idx = text.find(end_marker, start_idx)
        if end_idx == -1:
            # No closing marker, return from start to end
            return text[start_idx:].strip()
        
        # Extract code between markers
        code = text[start_idx:end_idx].strip()
        
        # Remove any remaining markers that might be inside
        code = code.replace(start_marker, '')
        code = code.replace(end_marker, '')
        
        return code

    def _groq_api(self, prompt: str) -> str:
        """Generate code using Groq Cloud API"""
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer <GROQ_API_KEY>",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        time.sleep(10)
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return self._extract_code(content)
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return ""
    def _deepseek_api(self, prompt: str) -> str:
        """Generate code using DeepSeek API with added debugging"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer <DEEPSEEK_API_KEY>",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        logger.debug(f"Sending request to DeepSeek API with payload:\n{json.dumps(payload, indent=2)}")
        logger.debug(f"Prompt content (first 200 chars): {prompt[:200]}...")
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            elapsed = time.time() - start_time
            
            logger.debug(f"API response received in {elapsed:.2f}s")
            logger.debug(f"Status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            response.raise_for_status()
            
            content = response.json()
            logger.debug(f"Full API response (truncated):\n{json.dumps(content, indent=2)[:500]}...")
            
            if "choices" not in content or len(content["choices"]) == 0:
                logger.error("No choices in API response")
                return ""
                
            message_content = content["choices"][0]["message"]["content"]
            logger.debug(f"Extracted message content (first 200 chars): {message_content[:200]}...")
            
            # Additional debug for code extraction
            extracted_code = self._extract_code(message_content)
            logger.debug(f"Extracted code (first 200 chars): {extracted_code[:200]}...")
            
            return extracted_code
            
        except requests.exceptions.Timeout:
            logger.error("DeepSeek API request timed out after 120 seconds")
            return ""
        except requests.exceptions.HTTPError as e:
            logger.error(f"DeepSeek API HTTP error: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Error response: {e.response.text}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected DeepSeek API error: {str(e)}", exc_info=True)
            return ""
        
    def _deepseek_api(self, prompt: str) -> str:
        """Generate code using DeepSeek API"""
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer <DEEPSEEK_API_KEY>",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            return ""

def main():
    parser = argparse.ArgumentParser(description="Advanced bug reproduction code generation")
    parser.add_argument("--bug_id", required=True, help="Bug ID to reproduce")
    parser.add_argument("--model", default="llama3-8b", 
                       help="Model to use: llama3-8b, qwen3-8b, deepseek-r1-7b, llama3-70b, gpt-4.1, deepseek-r1-7b")
    parser.add_argument("--technique", default="zero_shot", 
                       help="Prompting technique: zero_shot, few_shot, cot, in_context")
    parser.add_argument("--examples", type=int, default=3, 
                       help="Number of examples for few-shot learning (default: 3)")
    args = parser.parse_args()
    
    # Read bug report
    bug_report_path = Path(f"./dataset/{args.bug_id}/bug_report/{args.bug_id}.txt")
    with open(bug_report_path, "r", encoding="utf-8") as f:
        bug_report = f.read()
    
    output_dir = Path(f"./dataset/{args.bug_id}/reproduction_code")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating with {args.model} using {args.technique}")
    generator = CodeGenerator(args.model)
    start_time = time.time()
    code = generator.generate(bug_report, args.technique, args.examples)
    elapsed = time.time() - start_time
    
    if code:
        output_path = output_dir / f"{args.model}_{args.technique}.py"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"Generated code in {elapsed:.1f}s → {output_path}")
    else:
        logger.error("Failed to generate code")

if __name__ == "__main__":
    main()