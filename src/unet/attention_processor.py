# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers
import xformers.ops
import numpy as np
import math

import cv2
from basicsr.utils import img2tensor, tensor2img


class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        video_length=None,
        iter_cur=0,
        save_kv=True,
        mode='drag',
        mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]

        query = attn.head_to_batch_dim(query)

        if attn.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Memory bank design
        if attn.updown == 'up' and not save_kv:
            if mode == "move":
                if iter_cur>=5: 
                    key_ref = attn.buffer_key[iter_cur].to('cuda', dtype=query.dtype)
                    value_ref = attn.buffer_value[iter_cur].to('cuda', dtype=query.dtype)
                    key = key_ref.repeat(2,1,1)
                    value = value_ref.repeat(2,1,1)
            elif mode == 'drag':
                if iter_cur>=5:
                    key_ref = attn.buffer_key[iter_cur].to('cuda', dtype=query.dtype).repeat(2,1,1)
                    value_ref = attn.buffer_value[iter_cur].to('cuda', dtype=query.dtype).repeat(2,1,1)
                    key = torch.cat([key, key_ref], dim=1)
                    value = torch.cat([value, value_ref], dim=1)
            elif mode == 'landmark':
                if iter_cur>=5:
                    key_ref = attn.buffer_key[iter_cur].to('cuda', dtype=query.dtype).repeat(2,1,1)
                    value_ref = attn.buffer_value[iter_cur].to('cuda', dtype=query.dtype).repeat(2,1,1)
                    key = torch.cat([key, key_ref], dim=1)
                    value = torch.cat([value, value_ref], dim=1)
            elif mode in ['appearance', 'paste']:
                if 35>=iter_cur>=0:
                    key_ref = attn.buffer_key[iter_cur].to('cuda', dtype=query.dtype)
                    value_ref = attn.buffer_value[iter_cur].to('cuda', dtype=query.dtype)
                    key_fg = key_ref[1:]
                    value_fg = value_ref[1:]
                    key_bg = key_ref[:1]
                    value_bg = value_ref[:1]
                    mask_fg = mask['replace']
                    scale = np.sqrt(mask_fg.shape[-1]*mask_fg.shape[-2]/value_fg.shape[1])
                    mask_fg = (mask_fg>0.5).float().to('cuda', dtype=query.dtype)
                    mask_fg = F.interpolate(mask_fg[None,None], (int(mask_fg.shape[-2]/scale), int(mask_fg.shape[-1]/scale)))[0].unsqueeze(-1)
                    mask_fg = mask_fg.reshape(1, -1, 1)>0.5
                    mask_bg = mask['base']
                    mask_bg = (mask_bg>0.5).float().to('cuda', dtype=query.dtype)
                    mask_bg = F.interpolate(mask_bg[None,None], (int(mask_bg.shape[-2]/scale), int(mask_bg.shape[-1]/scale)))[0].unsqueeze(-1)
                    mask_bg = mask_bg.reshape(1, -1, 1)<0.5
                    key_fg = key_fg[mask_fg.repeat(key_fg.shape[0],1,key_fg.shape[2])].reshape(key_fg.shape[0], -1, key_fg.shape[2]).repeat(2,1,1)
                    value_fg = value_fg[mask_fg.repeat(value_fg.shape[0],1,value_fg.shape[2])].reshape(value_fg.shape[0], -1, value_fg.shape[2]).repeat(2,1,1)

                    key_bg = key_bg[mask_bg.repeat(key_bg.shape[0],1,key_bg.shape[2])].reshape(key_bg.shape[0], -1, key_bg.shape[2]).repeat(2,1,1)
                    value_bg = value_bg[mask_bg.repeat(value_bg.shape[0],1,value_bg.shape[2])].reshape(value_bg.shape[0], -1, value_bg.shape[2]).repeat(2,1,1)
                    key = torch.cat([key_bg, key_fg], dim=1) 
                    value = torch.cat([value_bg, value_fg], dim=1)

        if attn.updown == 'up' and save_kv:
            if not hasattr(attn, 'buffer_key'):
                attn.buffer_key = {}
                attn.buffer_value = {}
            attn.buffer_key[iter_cur] = key.cpu()
            attn.buffer_value[iter_cur] = value.cpu()

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0)
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask
        )
        # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.step = 0

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        iter_cur = -1,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            if encoder_hidden_states.shape[1]<=77:
                ip_hidden_states = None
            else:
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for image prompt
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)
            if iter_cur==-1 or 10<=iter_cur<20: 
                hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        
        latents = self.latents.repeat(x.size(0), 1, 1)
        
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        return self.norm_out(latents)