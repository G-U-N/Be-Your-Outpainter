from einops import rearrange, repeat
from src.util import isinstance_str
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import math


class LoraBase(nn.Module):
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.adpater = nn.Parameter(torch.zeros(hidden_size))
        self.scale_adapter = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x=None):
        return self.adpater


class LoraLinear(nn.Module):
    def __init__(self, hidden_size, rank, output_size=None) -> None:
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(hidden_size, self.rank, bias=False)
        self.up = nn.Linear(
            self.rank, hidden_size if output_size is None else output_size, bias=False
        )
        self.k = 8
        nn.init.normal_(self.down.weight, std=1 / self.rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, train=True):
        if train:
            return self.k * self.up(self.down(x))
        else:
            hidden_feature = self.down(x)


class LoraD(nn.Module):
    def __init__(self, hidden_size, rank, rank_base=1):
        super().__init__()
        self.rank = rank
        self.lora = LoraLinear(hidden_size, self.rank)
        self.lora_base = LoraBase(hidden_size, None)

    def forward(self, x):
        return self.lora(x) + self.lora_base(x)


def register_motion_lora(unet):
    loras = nn.ModuleList([nn.ModuleList(), nn.ModuleList()])

    def set_forward_motion_lora(module, lora_to_q, lora_to_k, lora_to_v):

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            video_length=None,
        ):
            batch_size, sequence_length, _ = hidden_states.shape

            if module.attention_mode == "Temporal":
                d = hidden_states.shape[1]
                hidden_states = rearrange(
                    hidden_states, "(b f) d c -> (b d) f c", f=video_length
                )

                if module.pos_encoder is not None:
                    hidden_states = module.pos_encoder(hidden_states)

                encoder_hidden_states = (
                    repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                    if encoder_hidden_states is not None
                    else encoder_hidden_states
                )
            else:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states

            if module.group_norm is not None:
                hidden_states = module.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = module.to_q(hidden_states) + lora_to_q(hidden_states)
            dim = query.shape[-1]
            query = module.reshape_heads_to_batch_dim(query)

            if module.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )

            key = module.to_k(encoder_hidden_states) + lora_to_k(encoder_hidden_states)
            value = module.to_v(encoder_hidden_states) + lora_to_v(
                encoder_hidden_states
            )

            key = module.reshape_heads_to_batch_dim(key)
            value = module.reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(
                        attention_mask, (0, target_length), value=0.0
                    )
                    attention_mask = attention_mask.repeat_interleave(
                        module.heads, dim=0
                    )

            # attention, what we cannot get enough of
            if module._use_memory_efficient_attention_xformers:
                hidden_states = module._memory_efficient_attention_xformers(
                    query, key, value, attention_mask
                )
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if (
                    module._slice_size is None
                    or query.shape[0] // module._slice_size == 1
                ):
                    hidden_states = module._attention(query, key, value, attention_mask)
                else:
                    hidden_states = module._sliced_attention(
                        query, key, value, sequence_length, dim, attention_mask
                    )

            # linear proj
            hidden_states = module.to_out[0](hidden_states)

            # dropout
            hidden_states = module.to_out[1](hidden_states)

            if module.attention_mode == "Temporal":
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

            return hidden_states

        module.forward = forward
        return forward

    for name, module in unet.named_modules():

        if isinstance_str(module, "VersatileAttention"):
            hidden_size = module.to_q.weight.shape[0]
            to_q = LoraLinear(hidden_size, 16)
            to_v = LoraLinear(hidden_size, 16)
            to_k = LoraLinear(hidden_size, 16)

            loras[0].append(nn.ModuleList([to_q, to_k, to_v]))
            set_forward_motion_lora(module, to_q, to_k, to_v)

        if isinstance_str(module, "FeedForward") and "temporal_transformer" in name:
            print(name)
            output_size, input_size = module.net[0].proj.weight.shape
            lora_linear = LoraLinear(input_size, 16, output_size)
            module.net[0].proj.lora_linear = lora_linear
            loras[1].append(lora_linear)
            output_size, input_size = module.net[2].weight.shape
            lora_linear = LoraLinear(input_size, 16, output_size)
            module.net[2].lora_layer = lora_linear
            loras[1].append(lora_linear)

    return loras


def adjust_motion_lora(
    unet, loras, scale_l, scale_r, scale_t, scale_b, height, width, inference=True, k=1
):

    expand_l = int(scale_l * width)
    expand_r = int(scale_r * width)
    expand_t = int(scale_t * height)
    expand_b = int(scale_b * height)
    h_exp = height + expand_t + expand_b
    w_exp = width + expand_l + expand_r

    if inference and k == "inf":
        mask = torch.ones((h_exp, w_exp))
    # global mask
    elif inference:
        # Initialize mask
        mask = np.zeros((h_exp, w_exp))

        # Fill the expanded square with 1s
        mask[expand_t : expand_t + height, expand_l : expand_l + width] = 1

        # Calculate the center of the expanded square
        center_x, center_y = w_exp // 2, h_exp // 2

        # Fill values outside the expanded square
        for i in range(h_exp):
            for j in range(w_exp):

                if mask[i, j] == 0:

                    distance_to_top = abs(i - expand_t)
                    distance_to_bottom = abs(i - (expand_t + height - 1))

                    min_vertical = (
                        1e9
                        if i >= expand_t and i <= expand_t + height - 1
                        else min(distance_to_top, distance_to_bottom)
                    )

                    distance_to_left = abs(j - expand_l)
                    distance_to_right = abs(j - (expand_l + width - 1))

                    min_horizontal = (
                        1e9
                        if j >= expand_l and j <= expand_l + width - 1
                        else min(distance_to_left, distance_to_right)
                    )

                    min_distance = min(min_vertical, min_horizontal)

                    max_range = max(expand_t, expand_b, expand_l, expand_r)

                    # Decrease the mask value based on the distance
                    mask[i, j] = np.exp(
                        -min_distance / (k * max_range)
                    )  # You can adjust the 2.0 to control the rate of decrease

        mask = torch.from_numpy(mask)
    else:
        mask = torch.ones((height, width))

    def set_forward_motion_lora(module, lora_to_q, lora_to_k, lora_to_v, mask):

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            video_length=None,
        ):
            batch_size, sequence_length, _ = hidden_states.shape

            bz = batch_size // video_length
            scale_factor = math.sqrt((mask.shape[0] * mask.shape[1]) // sequence_length)
            curr_mask = (
                F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), scale_factor=1 / scale_factor
                )
                .squeeze(0)
                .squeeze(0)
            )
            curr_mask = rearrange(curr_mask, "h w -> (h w) 1  1")
            curr_mask = repeat(curr_mask, "hw 1 1 -> (k hw) 1 1", k=bz)
            curr_mask = curr_mask.to(hidden_states.device, hidden_states.dtype)

            if module.attention_mode == "Temporal":
                d = hidden_states.shape[1]
                hidden_states = rearrange(
                    hidden_states, "(b f) d c -> (b d) f c", f=video_length
                )

                if module.pos_encoder is not None:
                    hidden_states = module.pos_encoder(hidden_states)

                encoder_hidden_states = (
                    repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d)
                    if encoder_hidden_states is not None
                    else encoder_hidden_states
                )
            else:
                raise NotImplementedError

            encoder_hidden_states = encoder_hidden_states

            if module.group_norm is not None:
                hidden_states = module.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = module.to_q(hidden_states) + curr_mask * lora_to_q(hidden_states)

            dim = query.shape[-1]
            query = module.reshape_heads_to_batch_dim(query)

            if module.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )
            key = module.to_k(encoder_hidden_states) + curr_mask * lora_to_k(
                encoder_hidden_states
            )
            value = module.to_v(encoder_hidden_states) + curr_mask * lora_to_v(
                encoder_hidden_states
            )
            key = module.reshape_heads_to_batch_dim(key)
            value = module.reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(
                        attention_mask, (0, target_length), value=0.0
                    )
                    attention_mask = attention_mask.repeat_interleave(
                        module.heads, dim=0
                    )

            # attention, what we cannot get enough of
            if module._use_memory_efficient_attention_xformers:
                hidden_states = module._memory_efficient_attention_xformers(
                    query, key, value, attention_mask
                )
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if (
                    module._slice_size is None
                    or query.shape[0] // module._slice_size == 1
                ):
                    hidden_states = module._attention(query, key, value, attention_mask)
                else:
                    hidden_states = module._sliced_attention(
                        query, key, value, sequence_length, dim, attention_mask
                    )

            # linear proj
            hidden_states = module.to_out[0](hidden_states)

            # dropout
            hidden_states = module.to_out[1](hidden_states)

            if module.attention_mode == "Temporal":
                hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

            return hidden_states

        module.forward = forward
        return forward

    idx = 0
    for _, module in unet.named_modules():
        if isinstance_str(module, "VersatileAttention"):
            hidden_size = module.to_q.weight.shape[0]
            to_q, to_k, to_v = loras[0][idx]
            set_forward_motion_lora(module, to_q, to_k, to_v, mask)
            idx += 1
    return loras
