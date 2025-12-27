import torch
from x_transformers import AutoModelForSeq2SeqLM

def load_seq2seq_model(model_name: str = 't5-base') -> AutoModelForSeq2SeqLM:
    return AutoModelForSeq2SeqLM.from_pretrained(model_name, return_dict=True)

def prepare_inputs():
    # Initial decoder input token(s)
    init_tokens = torch.tensor([[0]])
    # Source/encoding sequences (3 x 2)
    src_sequences = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # Mask for the encoding sequence (3 x 1)
    enc_mask = torch.tensor([[False], [True], [False]])
    return init_tokens, src_sequences, enc_mask

def generate_from_model(model, init_tokens, src_sequences, enc_mask,
                        max_length: int = 7,
                        num_beams: int = 1,
                        early_stopping: bool = True):
    enc_len = len(src_sequences)
    return model.generate(
        start_tokens=init_tokens,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=early_stopping,
        enc_seq_len=enc_len,
        mask=enc_mask
    )

def main():
    seq2seq_model = load_seq2seq_model()
    # Retain accessors to encoder/decoder as in original code
    encoder = seq2seq_model.get_encoder()
    decoder = seq2seq_model.get_decoder()

    init_tokens, src_sequences, enc_mask = prepare_inputs()
    sample = generate_from_model(seq2seq_model, init_tokens, src_sequences, enc_mask)
    print(sample)

if __name__ == '__main__':
    main()