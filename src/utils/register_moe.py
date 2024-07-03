from einops import rearrange, repeat
from src.util import isinstance_str
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import cv2
import math
import matplotlib.pyplot as plt


def one_hot_to_rgb(one_hot_vector):
    colors = np.array(
        [[255, 0, 0], [0, 255, 0], [0, 0, 255], [127, 127, 0]]
    )  # 假设 d = 3
    rgb_image = np.dot(one_hot_vector, colors)
    return rgb_image.astype(np.uint8)


def save_video(tensor, video_filename):
    tensor = tensor.detach().cpu()
    b, h, w, f, d = tensor.shape
    tensor = tensor.reshape((b, h, w, f, d))
    tensor = tensor.permute(0, 3, 1, 2, 4)  # 调整维度以便于处理
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (w, h))

    for i in range(b):
        for j in range(f):
            one_hot_frame = tensor[i, j, :, :, :]
            rgb_frame = one_hot_to_rgb(one_hot_frame)
            out.write(rgb_frame)

    out.release()


im_id = 0


def plot_labels(square_array):
    global im_id
    plt.imshow(square_array, cmap="tab20", interpolation="nearest")
    plt.colorbar()
    plt.savefig(f"{im_id}.png")
    # im_id+=1
    plt.clf()


class LoraLinear(nn.Module):
    def __init__(self, hidden_size, rank) -> None:
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(hidden_size, self.rank, bias=False)
        self.up = nn.Linear(self.rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=1 / self.rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.down(x))


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), "Shape error"
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


class CosineLinear(nn.Module):
    def __init__(
        self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True
    ):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(
            F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1)
        )

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return out


class MoELora(nn.Module):
    def __init__(
        self, num_loras, hidden_size, rank, tau_start=1.0, tau_end=1 / 16, tau_steps=200
    ) -> None:
        super().__init__()
        self.num_loras = num_loras
        self.hidden_size = hidden_size
        self.rank = rank
        self.lora_experts = nn.ModuleList(
            [LoraLinear(hidden_size, rank) for _ in range(num_loras)]
        )
        self.gate = nn.Sequential(CosineLinear(16 * hidden_size, num_loras))
        self.status = "train"

        self.taus = np.linspace(tau_start, tau_end, tau_steps)
        self.tau_steps = tau_steps
        self.step = 0
        self.tau = 1

    def forward(self, x):
        if self.step < self.tau_steps and self.status == "train":
            self.tau = self.taus[self.step]
            self.step += 1

        logits = self.gate(rearrange(x, "b f c -> b (f c)"))
        gumbel_softmax = self.gumbel_softmax(logits)  # (b h w) x (f x d)
        if self.status == "train":
            global im_id
            im_id += 1
            if im_id % 500 == 0:
                h = int(np.sqrt(gumbel_softmax.shape[0]))
                gumbel_softmax_input = (
                    rearrange(
                        gumbel_softmax.detach().cpu(), "(1 h w) n -> 1 h w n", h=h, w=h
                    )
                    .squeeze(0)
                    .numpy()
                )

                gumbel_softmax_input = np.argmax(gumbel_softmax_input, axis=-1)
                print(
                    (gumbel_softmax_input == 0).mean(),
                    (gumbel_softmax_input == 1).mean(),
                    (gumbel_softmax_input == 2).mean(),
                    (gumbel_softmax_input == 3).mean(),
                )
                plot_labels(gumbel_softmax_input)

        gumbel_softmax = gumbel_softmax.unsqueeze(1)
        outputs = torch.cat(
            [expert(x).unsqueeze(2) for expert in self.lora_experts], dim=2
        )  # b x f x num x d
        weighted_outputs = gumbel_softmax.unsqueeze(-1) * outputs
        output = weighted_outputs.sum(dim=2)
        return output

    def gumbel_softmax(self, logits):
        U = torch.rand_like(logits)
        G = -torch.log(-torch.log(U + 1e-12) + 1e-12)
        y_soft = F.softmax((logits + G) / self.tau, dim=-1)

        if self.status == "train":
            _, y_hard = y_soft.max(dim=-1)
            y_hard = F.one_hot(y_hard, num_classes=logits.size(-1))
            y_hard = y_hard.type(logits.type())
            print((y_soft - y_hard).abs().mean())
            return y_hard + (y_soft - y_soft.detach()).clone()

        elif self.status == "eval":
            _, y_hard = y_soft.max(dim=-1)
            y_hard = F.one_hot(y_hard, num_classes=logits.size(-1))
            y_hard = y_hard.type(logits.type())
            print((y_soft - y_hard).abs().mean())
            return y_hard + (y_soft - y_soft.detach()).clone()

        else:
            assert 0, "Not implemented yet"


def register_motion_lora(unet):
    loras = nn.ModuleList()

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
            to_q = MoELora(4, hidden_size, 1)
            to_k = LoraLinear(hidden_size, 4)
            to_v = LoraLinear(hidden_size, 4)
            loras.append(nn.ModuleList([to_q, to_v, to_k]))
            set_forward_motion_lora(module, to_q, to_k, to_v)
    return loras


def adjust_motion_lora(
    unet, loras, scale_l, scale_r, scale_t, scale_b, height, width, inference=True
):
    expand_l = int(scale_l * width)
    expand_r = int(scale_r * width)
    expand_t = int(scale_t * height)
    expand_b = int(scale_b * height)
    # Calculate expanded square dimensions
    h_exp = height + expand_t + expand_b
    w_exp = width + expand_l + expand_r

    if inference:
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
                    # Calculate distance to the nearest side of the original square
                    distance_to_top = abs(i - expand_t)
                    distance_to_bottom = abs(i - (expand_t + height - 1))
                    distance_to_left = abs(j - expand_l)
                    distance_to_right = abs(j - (expand_l + width - 1))

                    min_distance = min(
                        distance_to_top,
                        distance_to_bottom,
                        distance_to_left,
                        distance_to_right,
                    )

                    max_range = max(expand_t, expand_b, expand_l, expand_r)

                    # Decrease the mask value based on the distance
                    mask[i, j] = np.exp(
                        -min_distance / max_range
                    )  # You can adjust the 2.0 to control the rate of decrease

        mask = torch.from_numpy(mask)
    else:
        mask = torch.ones((h_exp, w_exp))

    print(mask)

    def set_forward_motion_lora(module, lora_to_q, lora_to_k, lora_to_v):

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            video_length=None,
        ):
            batch_size, sequence_length, _ = hidden_states.shape

            bz = batch_size // video_length
            scale_factor = (h_exp * w_exp) // sequence_length
            mask = (
                F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0), scale_factor=1 / scale_factor
                )
                .squeeze(0)
                .squeeze(0)
            )
            mask = rearrange(mask, "h w -> (h w) 1  1")
            mask = repeat(mask, "hw 1 1 -> (k hw) 1 1", k=bz)
            mask = mask.to(hidden_states.device, hidden_states.dtype)

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

            query = module.to_q(hidden_states) + mask * lora_to_q(hidden_states)
            dim = query.shape[-1]
            query = module.reshape_heads_to_batch_dim(query)

            if module.added_kv_proj_dim is not None:
                raise NotImplementedError

            encoder_hidden_states = (
                encoder_hidden_states
                if encoder_hidden_states is not None
                else hidden_states
            )

            key = module.to_k(encoder_hidden_states) + mask * lora_to_k(
                encoder_hidden_states
            )
            value = module.to_v(encoder_hidden_states) + mask * lora_to_v(
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
            to_q, to_k, to_v = loras[idx]
            if inference:
                for lora in loras[idx]:
                    if isinstance(lora, MoELora):
                        lora.status = "eval"
            else:
                for lora in loras[idx]:
                    if isinstance(lora, MoELora):
                        lora.status = "train"
            set_forward_motion_lora(module, to_q, to_k, to_v)
            idx += 1
    return loras
