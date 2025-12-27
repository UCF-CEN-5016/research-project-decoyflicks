import torch
from x_transformers import TransformerWrapper, AttentionLayers, Decoder

torch.manual_seed(42)

batch_size = 8
input_dim = (8, 1024)
x = torch.randint(0, 32, (batch_size, 1024))

attn_layers = AttentionLayers(
    dim=512,
    depth=4,
    heads=4,
    causal=True,
    cross_attend=False,
    only_cross=False,
    use_scalenorm=False,
    use_rmsnorm=False,
    use_simple_rmsnorm=False,
    alibi_pos_bias=False,
    alibi_num_heads=None,
    rel_pos_bias=False,
    rel_pos_num_buckets=32,
    rel_pos_max_distance=128,
    dynamic_pos_bias=False,
    dynamic_pos_bias_log_distance=False,
    dynamic_pos_bias_mlp_depth=2,
    dynamic_pos_bias_norm=False,
    rotary_pos_emb=True,
    rotary_emb_dim=None,
    rotary_xpos=False,
    rotary_interpolation_factor=1.0,
    rotary_xpos_scale_base=512,
    rotary_base_rescale_factor=1.0,
    custom_layers=None,
    sandwich_coef=None,
    par_ratio=None,
    weight_tie_layers=False,
    layers_execute_order=None,
    residual_attn=False,
    cross_residual_attn=False,
    macaron=False,
    pre_norm=True,
    pre_norm_has_final_norm=True,
    gate_residual=False,
    scale_residual=False,
    scale_residual_constant=1.0,
    shift_tokens=0,
    sandwich_norm=False,
    resi_dual=False,
    resi_dual_scale=1.0,
    layer_dropout=0.0,
    cross_attn_tokens_dropout=0.0,
    disable_abs_pos_emb=None,
    attn_num_mem_kv=20,
    attn_one_kv_head=True
)

model = TransformerWrapper(
    num_tokens=32,
    max_seq_len=0,
    attn_layers=attn_layers,
    emb_dim=None,
    max_mem_len=0,
    shift_mem_down=0,
    emb_dropout=0.0,
    post_emb_norm=False,
    num_memory_tokens=20,
    memory_tokens_interspersed_every=None,
    tie_embedding=False,
    logits_dim=None,
    use_abs_pos_emb=True,
    scaled_sinu_pos_emb=False,
    l2norm_embed=False,
    emb_frac_gradient=1.0,
    attn_z_loss_weight=1e-4
)

logits = model(x)

print("Input shape:", x.shape, "Input dtype:", x.dtype)
print("Output shape:", logits.shape, "Output dtype:", logits.dtype)
assert not torch.isnan(logits).any()