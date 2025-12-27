import torch
from domino.optimizer.optimizer import MixedPrecisionOptimizer
from domino.optimizer.clip_grads import clip_grad_norm_fp32

batch_size = 2
model_dim = 768
params = torch.randn(batch_size, model_dim, requires_grad=True).cuda()
grads = torch.randn(batch_size, model_dim).cuda()
optimizer = torch.optim.Adam([params], lr=0.001)

mixed_precision_optimizer = MixedPrecisionOptimizer(
    optimizer,
    clip_grad=0.0,
    log_num_zeros_in_grad=False,
    params_have_main_grad=True,
    use_contiguous_buffers_in_local_ddp=False,
    fp16=False,
    bf16=False,
    params_dtype=torch.float32,
    grad_scaler=None,
    models=[params]
)

global cdb
cdb = None

class Args:
    sequence_parallel = True

try:
    mixed_precision_optimizer.allreduce_layernorm_grads(Args())
except AttributeError as e:
    print(e)