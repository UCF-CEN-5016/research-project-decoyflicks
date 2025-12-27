import torch

oov_token = "<oov>"
text = ["こんにちは", "世界", oov_token, "テスト"]

def filter_oov(tokens, oov_token):
    return [t for t in tokens if t != oov_token]

print("text before filtering:", text)
filtered = filter_oov(text, oov_token)
print("text after filtering OOV:", filtered)