import torch
from typing import Any, Optional

class MLMGenerator:
    def __init__(
        self,
        net: Any,
        mask_id: int,
        no_replace_prob: float = 0.15,
        random_token_prob: float = 0.10,
        max_seq_len: int = 512,
        steps: int = 8,
        schedule_fn: Any = None,
        can_mask_prev_unmasked: bool = False,
        self_cond: bool = True,
        self_cond_train_prob: float = 0.75,
        token_critic: Optional[Any] = None,
        critic_loss_weight: float = 1.0
    ):
        assert not (token_critic and exists(token_critic))
        
        self.net = net
        self.dim = net.emb_dim
        self.num_tokens = net.num_tokens
        self.mask_id = mask_id
        self.no_replace_prob = no_replace_prob
        self.random_token_prob = random_token_prob
        self.max_seq_len = max_seq_len
        self.steps = steps
        
        if callable(schedule_fn):
            self.schedule_fn = schedule_fn
        elif schedule_fn == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule_fn == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule_fn}')
        
        self.can_mask_prev_unmasked = can_mask_prev_unmasked
        
        self.self_cond = self_cond
        if self_cond:
            self.null_embed = nn.Parameter(torch.randn(self.dim))
            self.to_self_cond = nn.Linear(self.dim, self.dim, bias=False) if self_cond else None
            self.self_cond_train_prob = self_cond_train_prob
        
        self.token_critic = token_critic
        if token_critic:
            self.token_critic = SelfCritic(net)
        
        self.critic_loss_weight = critic_loss_weight

    def generate(
        self,
        batch_size: Optional[int] = None,
        start_temperature: float = 1.0,
        filter_thres: Optional[float] = 0.7,
        noise_level_scale: float = 1.0,
        **kwargs: Any
    ) -> torch.Tensor:
        sample_one = not exists(batch_size)
        batch_size = default(batch_size, 1)
        
        device = next(self.net.parameters()).device
        
        was_training = self.training
        self.eval()
        
        times = torch.linspace(0., 1., self.steps + 1)
        
        shape = (batch_size, self.max_seq_len)
        seq = torch.full(shape, self.mask_id, device=device)
        mask = torch.full(shape, True, device=device)
        
        all_mask_num_tokens = (self.schedule_fn(times[1:]) * self.max_seq_len).long()
        
        has_self_cond = self.self_cond
        last_embed = self.null_embed if has_self_cond else None
        
        for mask_num_tokens, steps_until_x0 in zip(all_mask_num_tokens.tolist(), reversed(range(self.steps))):
            self_cond = self.to_self_cond(last_embed) if has_self_cond else None
            
            logits, embeds = self.net(
                seq,
                sum_embeds=self_cond,
                return_logits_and_embeddings=True,
                **kwargs
            )
            
            if has_self_cond:
                last_embed = embeds
            
            if exists(filter_thres):
                logits = top_k(logits, filter_thres)
            
            annealing_scale = steps_until_x0 / self.steps
            temperature = start_temperature * annealing_scale
            
            probs = (logits / max(temperature, 1e-3)).softmax(dim=1)
            
            sampled_ids = gumbel_sample(logits, temperature=max(temperature, 1e-3))
            
            seq = torch.where(mask, sampled_ids, seq)
            
            if exists(self.token_critic):
                scores = self.token_critic(seq)
                scores = rearrange(scores, 'b n 1 -> b n')
                scores = scores + noise_level_scale * gumbel_noise(scores) * annealing_scale
            else:
                scores = 1 - logits.softmax(dim=1)
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')
            
            if mask_num_tokens == 0:
                pass
            
            if not self.can_mask_prev_unmasked:
                scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
            
            mask_indices = scores.topk(mask_num_tokens, dim=1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, mask_indices, True)
            seq = seq.masked_fill(mask, self.mask_id)
        
        self.train(was_training)
        
        if sample_one:
            seq = rearrange(seq, '1 n -> n')
        
        return seq

    def forward(
        self,
        x: torch.Tensor,
        only_train_generator: bool = False,
        only_train_critic: bool = False,
        generator_sample_temperature: Optional[float] = None,
        **kwargs: Any
    ) -> Losses:
        b, n, device = *x.shape, x.device
        assert n == self.max_seq_len
        
        orig_seq = x.clone()
        
        rand_times = torch.empty(b, device=device).uniform_(0, 1)
        batched_randperm = torch.rand((b, n), device=device).argsort(dim=1).float()
        
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
        
        loss = F.cross_entropy(
            logits[mask],
            orig_seq[mask]
        )
        
        if not exists(self.token_critic) or only_train_generator:
            return Losses(loss, loss, None)
        
        sampled_ids = gumbel_sample(logits, temperature=default(generator_sample_temperature, random()))
        generated = torch.where(mask, sampled_ids, orig_seq)
        
        critic_logits = self.token_critic(generated)
        critic_labels = (sampled_ids != orig_seq).float()
        
        critic_loss = F.binary_cross_entropy_with_logits(
            rearrange(critic_logits, '... 1 -> ...'),
            critic_labels
        )
        
        if only_train_critic:
            total_loss = critic_loss
            loss = None
        else:
            total_loss = loss + critic_loss * self.critic_loss_weight
        
        return Losses(total_loss, loss, critic_loss)