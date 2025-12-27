import torch
from transformers import TransformerWrapper, Decoder

NUM_TOKENS = 32
MAX_SEQ_LEN = 0
NUM_MEMORY_TOKENS = 20
BATCH_SIZE = 8
SEQ_LENGTH = 1024


def build_decoder():
    return Decoder(
        dim=512,
        depth=4,
        heads=4,
        rotary_pos_emb=True,
        attn_flash=True,
        attn_onnxable=True,
        attn_num_mem_kv=NUM_MEMORY_TOKENS,
        attn_one_kv_head=True,
    )


def build_model():
    decoder = build_decoder()
    return TransformerWrapper(
        num_tokens=NUM_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        num_memory_tokens=NUM_MEMORY_TOKENS,
        attn_layers=decoder,
    )


def run_inference():
    model = build_model()
    tokens = torch.randint(0, NUM_TOKENS, (BATCH_SIZE, SEQ_LENGTH))
    logits = model(tokens)
    print(tokens.shape, tokens.dtype)
    print(logits.shape, logits.dtype)


if __name__ == "__main__":
    run_inference()