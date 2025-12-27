import torch
import deepspeed
from typing import Any, Dict, Tuple


class SimpleModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def print_throughput(wrapped_model: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
    """
    Helper that prints a simple throughput/debug message about the provided model.
    Accepts arbitrary additional arguments (e.g., Ellipsis) to remain compatible
    with previous call sites.
    """
    model_name = wrapped_model.__class__.__name__ if wrapped_model is not None else "None"
    param_count = sum(p.numel() for p in wrapped_model.parameters()) if wrapped_model is not None else 0
    print(f"Model: {model_name}, params: {param_count}, extra_args: {args} {kwargs}")


def build_model() -> torch.nn.Module:
    return SimpleModel()


def initialize_deepspeed_engine(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, config: Dict
) -> Tuple[Any, Any]:
    """
    Initialize and return the DeepSpeed engine and the (potentially replaced) optimizer.
    """
    engine, opt, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
    )
    return engine, opt


def train_one_epoch(engine: Any, batch_size: int = 2) -> None:
    """
    Run a single training epoch (kept to one iteration here to match original logic).
    """
    for _ in range(1):  # Train only one epoch / iteration
        inputs = torch.randn(batch_size, 10)
        outputs = engine(inputs)
        loss = outputs.sum()
        engine.backward(loss)
        engine.step()

        # Correct model access without causing an error (engine.module is the wrapped model)
        print_throughput(engine.module, ...)

    # Correct access to the wrapped model outside the loop
    print_throughput(engine.module, ...)


def main() -> None:
    base_model = build_model()
    base_optimizer = torch.optim.Adam(base_model.parameters())

    ds_config = {"train_batch_size": 2, "fp16": {"enabled": True}}

    engine, engine_optimizer = initialize_deepspeed_engine(
        model=base_model, optimizer=base_optimizer, config=ds_config
    )

    train_one_epoch(engine)


if __name__ == "__main__":
    main()