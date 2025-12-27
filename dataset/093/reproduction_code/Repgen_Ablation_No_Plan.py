import torch

def linear_schedule(t):
    return t

def cosine_schedule(t):
    return 0.5 * (1 + torch.cos(torch.pi * t))

def top_k(logits, k):
    v, _ = logits.topk(k)
    mask = logits >= v[:, [-1]]
    return logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

def gumbel_sample(logits, temperature=1.0):
    u = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = -torch.log(-torch.log(u))
    return torch.argmax(logits / temperature + gumbel_noise, dim=-1)

def sample_prob(p):
    return torch.rand(1) < p

def get_mask_subset_prob(mask, prob):
    num_tokens_to_keep = int(prob * mask.sum())
    indices = torch.where(mask)
    sampled_indices = torch.randperm(indices[0].numel())[:num_tokens_to_keep]
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask[indices[0][sampled_indices], indices[1][sampled_indices]] = True
    return new_mask

def generate(self, batch_size=None, start_temperature=1., filter_thres=0.7, noise_level_scale=1., **kwargs):
    sample_one = not exists(batch_size)
    batch_size = default(batch_size, 1)

    device = next(self.net.parameters()).device

    was_training = self.training
    self.eval()

    times = torch.linspace(0., 1., self.steps + 1)

    # sequence starts off as all masked
    shape = (batch_size, self.max_seq_len)
    seq = torch.full(shape, self.mask_id, device=device)
    mask = torch.full(shape, True, device=device)

    # slowly demask
    all_mask_num_tokens = (self.schedule_fn(times[1:]) * self.max_seq_len).long()

    has_self_cond = self.self_cond
    last_embed = self.null_embed if has_self_cond else None

    for mask_num_tokens, steps_until_x0 in zip(all_mask_num_tokens.tolist(), reversed(range(self.steps))):
        self_cond = self.to_self_cond(last_embed) if has_self_cond else None
        logits, embeds = self.net(seq, sum_embeds=self_cond, return_logits_and_embeddings=True, **kwargs)
        if has_self_cond:
            last_embed = embeds

        if exists(filter_thres):
            logits = top_k(logits, filter_thres)

        annealing_scale = steps_until_x0 / self.steps
        temperature = start_temperature * annealing_scale
        probs = (logits / max(temperature, 1e-3)).softmax(dim=-1)
        sampled_ids = gumbel_sample(logits, temperature=temperature)
        seq = torch.where(mask, sampled_ids, seq)

        if exists(self.token_critic):
            scores = self.token_critic(seq)
            scores = rearrange(scores, 'b n 1 -> b n')
            scores = scores + noise_level_scale * gumbel_noise(scores) * annealing_scale
        else:
            scores = 1 - logits.softmax(dim=-1)
            scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
            scores = rearrange(scores, 'b n 1 -> b n')

        if mask_num_tokens == 0:
            pass

        if not self.can_mask_prev_unmasked:
            scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
        mask_indices = scores.topk(mask_num_tokens, dim=-1).indices
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, mask_indices, True)
        seq = seq.masked_fill(mask, self.mask_id)

    self.train(was_training)
    if sample_one:
        seq = rearrange(seq, '1 n -> n')
    return seq

def forward(self, x, only_train_generator=False, only_train_critic=False, generator_sample_temperature=None, **kwargs):
    b, n, device = *x.shape, x.device
    assert n == self.max_seq_len
    orig_seq = x.clone()
    rand_times = torch.empty(b, device=device).uniform_(0, 1)
    batched_randperm = torch.rand((b, n), device=device).argsort(dim=-1).float()
    rand_probs = self.schedule_fn(rand_times)
    num_tokens_mask = (rand_probs * n).clamp(min=1.)
    mask = batched_randperm < rearrange(num_tokens_mask, 'b -> b 1')
    replace_mask_id_mask = mask.clone()
    frac_seq_left = 1.
    if self.no_replace_prob > 0. and coin_flip():
        frac_seq_left -= self.no_replace_prob
        no_replace_prob_mask = get_mask_subset_prob(mask, self.no_replace_prob)
        replace_mask_id_mask &= ~no_replace_prob_mask
    if self.random_token_prob > 0. and coin_flip():
        random_token_prob_mask = get_mask_subset_prob(replace_mask_id_mask, self.random_token_prob * frac_seq_left)
        random_tokens = torch.randint(0, self.num_tokens, (b, n), device=device)
        x = torch.where(random_token_prob_mask, random_tokens, x)
        replace_mask_id_mask &= ~random_token_prob_mask
    masked = torch.where(replace_mask_id_mask, self.mask_id, x)
    if self.self_cond:
        self_cond = self.null_embed
        if sample_prob(self.self_cond_train_prob):
            with torch.no_grad():
                self_cond = self.net(masked, return_embeddings=True, **kwargs).detach()
        kwargs.update(sum_embeds=self.to_self_cond(self_cond))
    context = torch.no_grad if only_train_critic else nullcontext
    with context():
        logits = self.net(masked, **kwargs)
    loss = F.cross_entropy(logits[mask], orig_seq[mask])
    if not exists(self.token_critic) or only_train_generator:
        return Losses(loss, loss, None)
    sampled_ids = gumbel_sample(logits, temperature=default(generator_sample_temperature, random()))
    generated = torch.where(mask, sampled_ids, orig_seq)
    critic_logits = self.token_critic(generated)
    critic_labels = (sampled_ids != orig_seq).float()
    critic_loss = F.binary_cross_entropy_with_logits(rearrange(critic_logits, '... 1 -> ...'), critic_labels)
    if only_train_critic:
        total_loss = critic_loss
        loss = None
    else:
        total_loss = loss + critic_loss * self.critic_loss_weight
    return Losses(total_loss, loss, critic_loss)