import torch
from x_transformers import AutoConfig, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "x-transformers/MiniLMv2-Mini"

config = AutoConfig.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, config=config)
model.eval()
model.to(device)

batch_size = 1
src_len = 256
tgt_len = 200

src = torch.randint(0, 1000, (batch_size, src_len))
src_mask = torch.ones((batch_size, src_len), dtype=torch.bool)
start_tokens = torch.randint(0, 1000, (batch_size, 1))

output = model.generate(src, start_tokens, src_len, mask=src_mask)