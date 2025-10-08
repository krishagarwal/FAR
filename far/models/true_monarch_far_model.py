import math
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import FluxPosEmbed, LabelEmbedding, TimestepEmbedding, Timesteps, apply_rotary_emb
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import LayerNorm, RMSNorm
from diffusers.utils import is_torch_version
from einops import rearrange

from far.utils.registry import MODEL_REGISTRY


class AdaLayerNormContinuous(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type='layer_norm',
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == 'layer_norm':
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == 'rms_norm':
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f'unknown norm_type {norm_type}')

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=-1)

        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type='layer_norm', bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == 'layer_norm':
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))

        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa

def proxy_apply_rotary_emb(x, rotary_emb):
    return apply_rotary_emb(x.transpose(1, 2), rotary_emb).transpose(1, 2)

class MonarchAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, num_iters=1, eps=1e-6):
        super().__init__()
        
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.input_dim = dim
        self.num_heads = num_heads
        self.sm_scale_sqrt = self.head_dim ** -0.25
        self.num_iters = num_iters

        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)
    
    def _forward_inference(self, x: torch.Tensor, image_rotary_emb, attention_mask, layer_kv_cache, eps=1e-6):
        k_cache = layer_kv_cache['kv_cache'].get('k_cache', None)
        v_cache = layer_kv_cache['kv_cache'].get('v_cache', None)

        batch_size = x.size(0)

        q = self.to_q(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.to_k(x).view(batch_size, -1, self.num_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        attention_mask = attention_mask[:, -q.shape[1]:, :] if attention_mask is not None else None
        image_rotary_emb = (image_rotary_emb[0][-q.shape[1]:, :], image_rotary_emb[1][-q.shape[1]:, :])

        q = proxy_apply_rotary_emb(q, image_rotary_emb) * self.sm_scale_sqrt
        k = proxy_apply_rotary_emb(k, image_rotary_emb) * self.sm_scale_sqrt

        block_b1, block_b2 = 8, 8

        q = q.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_dim) # (b, f, i, j, h, d)
        k = k.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_dim) # (b, f, k, l, h, d)
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=-5)
        v = self.to_v(x).view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_dim) # (b, f, k, l, h, d)
        if v_cache is not None:
            v = torch.cat([v_cache, v], dim=-5)
        
        outputs = []
        for f in range(q.size(1)):
            curr_q = q[:, f] # (b, i, j, h, d)
            kv_end = k.size(1) - q.size(1) + f + 1
            curr_k = k[:, :kv_end] # (b, f, k, l, h, d)
            curr_v = v[:, :kv_end] # (b, f, k, l, h, d)
            frames = curr_k.size(1)

            L = torch.eye(block_b1, device=q.device, dtype=q.dtype).view(1, 1, 1, 1, block_b1, block_b1).expand(batch_size, q.size(-2), frames, block_b2, block_b1, block_b1) # (b, h, f, j, k, i)

            for _ in range(self.num_iters):
                aR = torch.einsum("bhfjki,bijhd->bfkjhd", L, curr_q)
                bR = torch.einsum("bfkjhd,bfklhd->bhfkjl", aR, curr_k)
                cR = torch.einsum("bhfjki->bhfkj", L).unsqueeze(-1)
                R = torch.softmax(bR / (cR + eps), dim=-1)

                aL = torch.einsum("bhfkjl,bfklhd->bfjkhd", R, curr_k)
                bL = torch.einsum("bfjkhd,bijhd->bhfjki", aL, curr_q)
                cL = torch.einsum("bhfkji->bhfjk", torch.xlogy(R, R)).unsqueeze(-1)
                # softmax over f and k
                L = rearrange(bL - cL, 'b h f j k i -> b h j i (f k)')
                L = torch.softmax(L, dim=-1)
                L = rearrange(L, 'b h j i (f k) -> b h f j k i', f=frames, k=block_b1)

            out = torch.einsum("bhfkjl,bfklhd->bfjkhd", R, curr_v)
            out = torch.einsum("bhfjki,bfjkhd->bijhd", L, out)
            outputs.append(out)
        out = torch.stack(outputs, dim=1) # (b, f, i, j, h, d)
        out = out.reshape(batch_size, -1, self.num_heads, self.head_dim)

        if layer_kv_cache["is_cache_step"]:
            layer_kv_cache['kv_cache'] = {
                'k_cache': k[:, :(-layer_kv_cache['token_per_frame'] // 64)],
                'v_cache': v[:, :(-layer_kv_cache['token_per_frame'] // 64)],
            }

        return out.reshape(batch_size, -1, self.input_dim), layer_kv_cache

    def _forward_train(self, x: torch.Tensor, image_rotary_emb, attention_mask, layer_kv_cache, eps=1e-6):
        assert layer_kv_cache is None or layer_kv_cache['kv_cache'] is None, f"{layer_kv_cache['kv_cache'].keys()}"
        batch_size = x.size(0)

        q = self.to_q(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.to_k(x).view(batch_size, -1, self.num_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        attention_mask = attention_mask[:, -q.shape[1]:, :] if attention_mask is not None else None
        image_rotary_emb = (image_rotary_emb[0][-q.shape[1]:, :], image_rotary_emb[1][-q.shape[1]:, :])

        q = proxy_apply_rotary_emb(q, image_rotary_emb) * self.sm_scale_sqrt
        k = proxy_apply_rotary_emb(k, image_rotary_emb) * self.sm_scale_sqrt

        block_b1, block_b2 = 8, 8

        q = q.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_dim) # (b, f, i, j, h, d)
        k = k.view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_dim) # (b, f, k, l, h, d)
        v = self.to_v(x).view(batch_size, -1, block_b1, block_b2, self.num_heads, self.head_dim) # (b, f, k, l, h, d)

        outputs = []
        for f in range(q.size(1)):
            curr_q = q[:, f] # (b, i, j, h, d)
            kv_end = k.size(1) - q.size(1) + f + 1
            curr_k = k[:, :kv_end] # (b, f, k, l, h, d)
            curr_v = v[:, :kv_end] # (b, f, k, l, h, d)
            frames = curr_k.size(1)

            L = torch.eye(block_b1, device=q.device, dtype=q.dtype).view(1, 1, 1, 1, block_b1, block_b1).expand(batch_size, q.size(-2), frames, block_b2, block_b1, block_b1) # (b, h, f, j, k, i)

            for _ in range(self.num_iters):
                aR = torch.einsum("bhfjki,bijhd->bfkjhd", L, curr_q)
                bR = torch.einsum("bfkjhd,bfklhd->bhfkjl", aR, curr_k)
                cR = torch.einsum("bhfjki->bhfkj", L).unsqueeze(-1)
                R = torch.softmax(bR / (cR + eps), dim=-1)

                aL = torch.einsum("bhfkjl,bfklhd->bfjkhd", R, curr_k)
                bL = torch.einsum("bfjkhd,bijhd->bhfjki", aL, curr_q)
                cL = torch.einsum("bhfkji->bhfjk", torch.xlogy(R, R)).unsqueeze(-1)
                # softmax over f and k
                L = rearrange(bL - cL, 'b h f j k i -> b h j i (f k)')
                L = torch.softmax(L, dim=-1)
                L = rearrange(L, 'b h j i (f k) -> b h f j k i', f=frames, k=block_b1)
            
            out = torch.einsum("bhfkjl,bfklhd->bfjkhd", R, curr_v)
            out = torch.einsum("bhfjki,bfjkhd->bijhd", L, out)
            outputs.append(out)
        out = torch.stack(outputs, dim=1) # (b, f, i, j, h, d)
        out = out.reshape(batch_size, -1, self.num_heads, self.head_dim)

        return out.reshape(batch_size, -1, self.input_dim), layer_kv_cache
    
    def forward(self, x: torch.Tensor, image_rotary_emb, attention_mask, layer_kv_cache=None):
        if self.training:
            return self._forward_train(x, image_rotary_emb, attention_mask, layer_kv_cache)
        else:
            return self._forward_inference(x, image_rotary_emb, attention_mask, layer_kv_cache)

class FAR_TransformerBlock(nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0, num_iters=1):
        super().__init__()

        assert dim == num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZeroSingle(dim)
        self.attn = MonarchAttention(
            dim,
            num_attention_heads,
            bias=True,
            eps=1e-6,
            num_iters=num_iters,
        )
        self.norm2 = AdaLayerNormZeroSingle(dim)

        self.mlp = FeedForward(dim=dim, dim_out=dim, activation_fn='gelu-approximate')

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        attention_mask=None,
        layer_kv_cache=None
    ):
        norm_hidden_states, gate = self.norm1(hidden_states, emb=temb)

        attn_output, layer_kv_cache = self.attn(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            layer_kv_cache=layer_kv_cache
        )
        hidden_states = hidden_states + gate * attn_output

        norm_hidden_states, gate = self.norm2(hidden_states, emb=temb)
        hidden_states = hidden_states + gate * self.mlp(norm_hidden_states)
        return hidden_states, layer_kv_cache


class FAR(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True
    _no_split_modules = ['FAR_TransformerBlock']

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 32,
        num_layers: int = 12,
        attention_head_dim: int = 64,
        num_attention_heads: int = 12,
        axes_dims_rope: Tuple[int] = (16, 24, 24),
        out_channels=32,
        slope_scale=0,
        short_term_ctx_winsize=16,
        num_iters=1,
        condition_cfg=None
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.x_embedder = torch.nn.Linear(self.config.in_channels * self.config.patch_size * self.config.patch_size, self.inner_dim)

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)

        if condition_cfg is not None:
            if condition_cfg['type'] == 'label':
                self.label_embedder = LabelEmbedding(condition_cfg['num_classes'], self.inner_dim, dropout_prob=0.1)
            elif condition_cfg['type'] == 'action':
                self.action_embedder = LabelEmbedding(condition_cfg['num_action_classes'], self.inner_dim, dropout_prob=0.1)
            else:
                raise NotImplementedError

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        self.transformer_blocks = nn.ModuleList([
            FAR_TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                num_iters=num_iters,
            ) for i in range(self.config.num_layers)
        ])

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.initialize_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = value

    def _pack_latent_sequence(self, latents, patch_size):
        batch_size, num_frames, channel, height, width = latents.shape
        height, width = height // patch_size, width // patch_size

        latents = rearrange(
            latents, 'b f c (h p1) (w p2) -> b (f h w) (c p1 p2)', b=batch_size, f=num_frames, c=channel, h=height, p1=patch_size, w=width, p2=patch_size)

        return latents

    def _prepare_latent_sequence_ids(self, batch_size, num_frames, height, width, patch_size, device, dtype):
        patch_size = self.config.patch_size
        height, width = height // patch_size, width // patch_size
        latent_image_ids = torch.zeros(num_frames, height, width, 3)

        latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(num_frames)[:, None, None]
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[None, :, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, None, :]

        latent_image_id_num_frames, latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(latent_image_id_num_frames * latent_image_id_height * latent_image_id_width, latent_image_id_channels)
        return latent_image_ids.to(device=device, dtype=dtype)

    def _unpack_latent_sequence(self, latents, num_frames, height, width):
        batch_size, num_patches, channels = latents.shape
        patch_size = self.config.patch_size
        height, width = height // patch_size, width // patch_size

        latents = latents.view(batch_size * num_frames, height, width, channels // (patch_size * patch_size), patch_size, patch_size)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, num_frames, channels // (patch_size * patch_size), height * patch_size, width * patch_size)
        return latents

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        if hasattr(self, 'label_embedder'):
            nn.init.normal_(self.label_embedder.embedding_table.weight, std=0.02)
        if hasattr(self, 'action_embedder'):
            nn.init.normal_(self.action_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.linear_2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm2.linear.weight, 0)
            nn.init.constant_(block.norm2.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def _build_causal_mask(self, input_shape, device, dtype):

        batch_size, num_frames, seq_len = input_shape
        token_per_frame = seq_len // num_frames

        def get_relative_positions(seq_len) -> torch.tensor:
            frame_idx = torch.arange(seq_len, device=device) // token_per_frame
            return (frame_idx.unsqueeze(0) - frame_idx.unsqueeze(1)).unsqueeze(0)

        # step 1: build context-context causal mask
        idx = torch.arange(seq_len, device=device)
        row_idx = idx.unsqueeze(1)  # (seq_len, 1)
        col_idx = idx.unsqueeze(0)  # (1, seq_len)
        # floor(i / N) >= floor(j / N)
        attention_mask = (row_idx // token_per_frame >= col_idx // token_per_frame).unsqueeze(0)

        attn_mask = torch.zeros(attention_mask.shape, device=device)
        attn_mask.masked_fill_(attention_mask.logical_not(), float('-inf'))

        linear_bias = self.config.slope_scale * get_relative_positions(seq_len)
        linear_bias.masked_fill_(attention_mask.logical_not(), 0)

        attn_mask += linear_bias
        return attn_mask.to(dtype)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor = None,
        context_cache={'kv_cache': None},
        conditions: torch.LongTensor = None,
        return_dict: bool = True,
    ):

        batch_size, num_frames, _, height, width = hidden_states.shape
        token_per_frame = (height // self.config.patch_size) * (width // self.config.patch_size)

        # step 1: pack latent sequence (and compute rope)
        hidden_states = self._pack_latent_sequence(hidden_states, patch_size=self.config.patch_size)
        latent_seq_ids = self._prepare_latent_sequence_ids(
            batch_size,
            num_frames,
            height,
            width,
            patch_size=self.config.patch_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype)

        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1).repeat((1, num_frames))

        if context_cache['kv_cache'] is not None:
            if context_cache['is_cache_step'] is True:
                # encode new context and current noise
                current_seq_len = hidden_states.shape[1] - context_cache['cached_seqlen']
                context_cache['cached_seqlen'] = hidden_states.shape[1] - token_per_frame

                hidden_states = hidden_states[:, -current_seq_len:, ...]
                timestep = timestep[:, -(current_seq_len // token_per_frame):]

                if self.config.condition_cfg is not None and self.config.condition_cfg['type'] == 'action':
                    conditions['action'] = conditions['action'][:, -(current_seq_len // token_per_frame):]
            else:
                # encode current noise
                hidden_states = hidden_states[:, -token_per_frame:, ...]
                timestep = timestep[:, -1:]
                if self.config.condition_cfg is not None and self.config.condition_cfg['type'] == 'action':
                    conditions['action'] = conditions['action'][:, -1:]

        # step 3: generate attention mask
        attention_mask = self._build_causal_mask(
            input_shape=(batch_size, num_frames, num_frames * token_per_frame),
            device=hidden_states.device,
            dtype=hidden_states.dtype)

        # step 4: input projection and linear embed + concat inputs
        hidden_states = self.x_embedder(hidden_states)
        seq_rotary_emb = self.pos_embed(latent_seq_ids)

        # noise timestep embedding
        timestep = rearrange(timestep, 'b t -> (b t)')
        timesteps_proj = self.time_proj(timestep.to(hidden_states.dtype))
        temb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)
        temb = rearrange(temb, '(b t) c -> b t c', b=batch_size).repeat_interleave(token_per_frame, dim=1)

        if self.config.condition_cfg is not None:
            if self.config.condition_cfg['type'] == 'label':
                label_emb = self.label_embedder(conditions['label']).unsqueeze(1)
                temb = temb + label_emb
            elif self.config.condition_cfg['type'] == 'action':
                action = rearrange(conditions['action'], 'b t -> (b t)')
                action_emb = self.action_embedder(action)
                action_emb = rearrange(action_emb, '(b t) c -> b t c', b=batch_size)
                action_emb = action_emb.repeat_interleave(token_per_frame, dim=1)
                temb = temb + action_emb
            else:
                raise NotImplementedError

        for index_block, block in enumerate(self.transformer_blocks):

            if context_cache['kv_cache'] is None:
                layer_kv_cache = {'kv_cache': None}
            elif index_block not in context_cache['kv_cache']:
                layer_kv_cache = {
                    'is_cache_step': context_cache['is_cache_step'],
                    'kv_cache': {},
                    'token_per_frame': token_per_frame
                }
            else:
                layer_kv_cache = {
                    'is_cache_step': context_cache['is_cache_step'],
                    'kv_cache': context_cache['kv_cache'][index_block],
                    'token_per_frame': token_per_frame
                }

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):

                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {
                    'use_reentrant': False
                } if is_torch_version('>=', '1.11.0') else {}
                hidden_states, layer_kv_cache = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    seq_rotary_emb,
                    attention_mask,
                    layer_kv_cache,
                    **ckpt_kwargs,
                )

            else:
                hidden_states, layer_kv_cache = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=seq_rotary_emb,
                    attention_mask=attention_mask,
                    layer_kv_cache=layer_kv_cache)

            if context_cache['kv_cache'] is not None:
                context_cache['kv_cache'][index_block] = layer_kv_cache['kv_cache']

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if context_cache['kv_cache'] is not None:
            output = output[:, -token_per_frame:, :]
            output = self._unpack_latent_sequence(output, num_frames=1, height=height, width=width)
        else:
            output = self._unpack_latent_sequence(output, num_frames=num_frames, height=height, width=width)
            if not self.training:
                output = output[:, -1:, ...]

        if not return_dict:
            return (output, context_cache)

        return SimpleNamespace(sample=output, context_cache=context_cache)


@MODEL_REGISTRY.register()
def FAR_B_TRUE_MONARCH_SLOW(**kwargs):
    return FAR(in_channels=32, out_channels=32, num_layers=12, attention_head_dim=64, patch_size=1, num_attention_heads=12, num_iters=8, **kwargs)


@MODEL_REGISTRY.register()
def FAR_M_TRUE_MONARCH_SLOW(**kwargs):
    return FAR(in_channels=32, out_channels=32, num_layers=12, attention_head_dim=64, patch_size=1, num_attention_heads=16, num_iters=8, **kwargs)


@MODEL_REGISTRY.register()
def FAR_L_TRUE_MONARCH_SLOW(**kwargs):
    return FAR(in_channels=32, out_channels=32, num_layers=24, attention_head_dim=64, patch_size=1, num_attention_heads=16, num_iters=8, **kwargs)


@MODEL_REGISTRY.register()
def FAR_XL_TRUE_MONARCH_SLOW(**kwargs):
    return FAR(in_channels=32, out_channels=32, num_layers=28, attention_head_dim=64, patch_size=1, num_attention_heads=18, num_iters=8, **kwargs)

@MODEL_REGISTRY.register()
def FAR_B_TRUE_MONARCH_FAST(**kwargs):
    return FAR(in_channels=32, out_channels=32, num_layers=12, attention_head_dim=64, patch_size=1, num_attention_heads=12, num_iters=1, **kwargs)


@MODEL_REGISTRY.register()
def FAR_M_TRUE_MONARCH_FAST(**kwargs):
    return FAR(in_channels=32, out_channels=32, num_layers=12, attention_head_dim=64, patch_size=1, num_attention_heads=16, num_iters=1, **kwargs)


@MODEL_REGISTRY.register()
def FAR_L_TRUE_MONARCH_FAST(**kwargs):
    return FAR(in_channels=32, out_channels=32, num_layers=24, attention_head_dim=64, patch_size=1, num_attention_heads=16, num_iters=1, **kwargs)


@MODEL_REGISTRY.register()
def FAR_XL_TRUE_MONARCH_FAST(**kwargs):
    return FAR(in_channels=32, out_channels=32, num_layers=28, attention_head_dim=64, patch_size=1, num_attention_heads=18, num_iters=1, **kwargs)