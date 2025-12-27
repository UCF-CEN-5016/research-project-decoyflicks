def align_right(x, pad_id, pad_len):
    # ...
    x = F.pad(x, (pad_len, 0), value=0)  # pad_id is ignored here
    return x