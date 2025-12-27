import torch
from transformers import ContinuousTransformerWrapper, Decoder
from typing import List, Tuple


def build_decoder() -> Decoder:
    return Decoder(
        dim=512,
        depth=6,
        heads=4,
        rotary_pos_emb=True,
        shift_tokens=1,
        attn_flash=True,
        attn_onnxable=True,
        use_rmsnorm=True,
        sandwich_norm=True,
    )


def build_model(decoder: Decoder) -> ContinuousTransformerWrapper:
    return ContinuousTransformerWrapper(
        dim_out=259,
        max_seq_len=0,
        num_memory_tokens=20,
        max_mem_len=100,
        attn_layers=decoder,
    )


def prepare_inputs(batch_size: int = 1, seq_len: int = 1024, dim: int = 512) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    inputs = torch.randn(batch_size, seq_len, dim)
    mask = torch.randn(batch_size, seq_len) > 0
    memories = [torch.zeros(batch_size, 100, dim) for _ in range(6)]
    return inputs, mask, memories


def run_inference(model: ContinuousTransformerWrapper, inputs: torch.Tensor, mask: torch.Tensor, memories: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    logits, new_memories = model(inputs, mask=mask, mems=memories, return_mems=True)
    return logits, new_memories


if __name__ == "__main__":
    decoder = build_decoder()
    model = build_model(decoder)

    inputs, mask, memories = prepare_inputs()
    logits, memories = run_inference(model, inputs, mask, memories)

    print(logits.shape)
    print([m.shape for m in memories])