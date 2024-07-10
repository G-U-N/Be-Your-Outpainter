# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py
# Adapted from https://github.com/G-U-N/Be-Your-Outpainter

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange
import copy
from src.models.pretrain.unet import UNet3DConditionModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class OutpaintPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class OutpaintPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    """
    OutPainting Prompts
    """

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_l=None,
        prompt_r=None,
        prompt_t=None,
        prompt_b=None,
    ):

        text_embedding = self._encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
        )

        embedding_l = (
            self._encode_prompt(
                prompt_l,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
            if prompt_l is not None
            else None
        )

        embedding_r = (
            self._encode_prompt(
                prompt_r,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
            if prompt_r is not None
            else None
        )

        embedding_t = (
            self._encode_prompt(
                prompt_t,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
            if prompt_t is not None
            else None
        )

        embedding_b = (
            self._encode_prompt(
                prompt_b,
                device,
                num_videos_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )
            if prompt_b is not None
            else None
        )

        return dict(
            text_embedding=text_embedding,
            embedding_l=embedding_l,
            embedding_r=embedding_r,
            embedding_t=embedding_t,
            embedding_b=embedding_b,
        )

    """
    VAE_ENCODING & VAE_DECODING
    """

    def _encode(self, videos):
        # videos [-1,1]
        video_length = videos.shape[2]
        videos = rearrange(videos, "b c f h w -> (b f) c h w")
        latents = []
        for frame_idx in range(videos.shape[0]):
            latents.append(
                self.vae.encode(videos[frame_idx : frame_idx + 1]).latent_dist.sample()
            )
        latents = torch.cat(latents)
        latents = (
            rearrange(latents, "(b f) c h w -> b c f h w", f=video_length) * 0.18215
        )
        return latents

    def _decode(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in range(latents.shape[0]):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        return video

    def decode_latents(self, latents):

        video = self._decode(latents)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def encode_videos(self, videos):
        v1 = videos
        # l1 = self._encode(v1)
        # v2 = self._decode(l1)
        # l2 = self._encode(v2)
        l1 = self._encode(v1)

        # latents = l1 + (l1-l2) # TODO: Helpful?

        latents = l1
        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        video_length,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // 8,
            width // 8,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                # shape = (1,) + shape[1:]
                latents = [
                    torch.randn(
                        shape, generator=generator[i], device=rand_device, dtype=dtype
                    )
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=rand_device, dtype=dtype
                ).to(device)
                # latents = torch.cat([latents]*video_length,dim=2)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )

            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def prepare_control_image(
        self,
        image,
        width,
        height,
        expand_l,
        expand_r,
        expand_t,
        expand_b,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):

        b, f, c, h, w = image.shape
        tmp = -torch.ones(b, f, c, h + expand_t + expand_b, w + expand_l + expand_r)
        tmp[..., expand_t : expand_t + height, expand_l : expand_l + width] = image
        image = tmp

        video_length = image.shape[2]
        image = rearrange(image, "b c f w h -> (b f) c w h")

        # image = self.control_image_processor.preprocess(image).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)
        image = rearrange(image, "(b f) c w h -> b c f w h", f=video_length)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def prepare_flow(self, bwd_flow, bwd_mask, fwd_flow, fwd_mask, device, dtype):

        bwd_flow = (
            torch.load(bwd_flow).to(device, dtype) if bwd_flow is not None else None
        )

        bwd_mask = (
            torch.load(bwd_mask).to(device, dtype) if bwd_mask is not None else None
        )

        fwd_flow = (
            torch.load(fwd_flow).to(device, dtype) if fwd_flow is not None else None
        )

        fwd_mask = (
            torch.load(fwd_mask).to(device, dtype) if fwd_mask is not None else None
        )

        return dict(
            bwd_flow=bwd_flow, bwd_mask=bwd_mask, fwd_flow=fwd_flow, fwd_mask=fwd_mask
        )

    """
    Core Function
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        scale_l: float,
        scale_r: float,
        scale_t: float,
        scale_b: float,
        height: int,
        width: int,
        video_length: int,
        window_size: int,
        stride: int,
        repeat_time: int,
        jump_length: int,
        init_video: torch.tensor,  # range: [-1,1]; shape: b c f h  w
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_l=None,
        prompt_r=None,
        prompt_t=None,
        prompt_b=None,
        bwd_mask=None,
        bwd_flow=None,
        fwd_mask=None,
        fwd_flow=None,
        warp_time=None,
        warp_step=None,
        is_grid=False,
        # Not Change
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        """

        Basic Params
        """

        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        expand_l = int(scale_l * width)
        expand_r = int(scale_r * width)
        expand_t = int(scale_t * height)
        expand_b = int(scale_b * height)
        assert (
            expand_l % 8 == 0
            and expand_r % 8 == 0
            and expand_t % 8 == 0
            and expand_b % 8 == 0
        )
        assert (height + expand_t + expand_b) % 8 == 0 and (
            width + expand_l + expand_r
        ) % 8 == 0
        assert height % 8 == 0 and width % 8 == 0

        latent_expand_l = expand_l // 8
        latent_expand_r = expand_r // 8
        latent_expand_t = expand_t // 8
        latent_expand_b = expand_b // 8

        latent_height = height // 8
        latent_width = width // 8

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        prompt_l = prompt_l if isinstance(prompt_l, list) else [prompt_l] * batch_size
        prompt_r = prompt_r if isinstance(prompt_r, list) else [prompt_r] * batch_size
        prompt_t = prompt_t if isinstance(prompt_t, list) else [prompt_t] * batch_size
        prompt_b = prompt_b if isinstance(prompt_b, list) else [prompt_b] * batch_size

        if negative_prompt is not None:
            negative_prompt = (
                negative_prompt
                if isinstance(negative_prompt, list)
                else [negative_prompt] * batch_size
            )

        text_embeddings = self.prepare_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_l=prompt_l,
            prompt_r=prompt_r,
            prompt_t=prompt_t,
            prompt_b=prompt_b,
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        """
        Latents
        """

        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height + expand_t + expand_b,
            width + expand_l + expand_r,
            text_embeddings["text_embedding"].dtype,
            device,
            generator,
            latents,
        )

        control = self.prepare_control_image(
            image=(init_video + 1) / 2,
            width=width,
            height=height,
            expand_l=expand_l,
            expand_r=expand_r,
            expand_t=expand_t,
            expand_b=expand_b,
            batch_size=batch_size * num_videos_per_prompt,
            num_images_per_prompt=num_videos_per_prompt,
            device=device,
            dtype=latents.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=False,
        )

        latents_dtype = latents.dtype
        noise = copy.deepcopy(latents)

        init_latents = self.encode_videos(init_video)

        flows = self.prepare_flow(
            bwd_flow,
            bwd_mask,
            fwd_flow,
            fwd_mask,
            device=latents.device,
            dtype=latents.dtype,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        i = 0
        k = 0
        j = 0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            while i < len(timesteps):
                values_tot = torch.zeros_like(latents)
                counts_tot = torch.zeros_like(latents)
                t = timesteps[i]
                seqs = get_views(video_length, window_size, stride)
                for idx, seq in enumerate(seqs):

                    latent_model_input = (
                        torch.cat([latents[:, :, seq]] * 2)
                        if do_classifier_free_guidance
                        else latents[:, :, seq]
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    control_input = control[:, :, seq]
                    # predict the noise residual
                    j = j + 1

                    if is_grid:
                        # TODO: FIX RIGHT = 0 OR BOTTOM = 0
                        values = torch.zeros_like(latent_model_input)
                        counts = torch.zeros_like(latent_model_input)
                        noise_pred_left = self.unet(
                            latent_model_input[..., :latent_width],
                            t,
                            encoder_hidden_states=text_embeddings["embedding_l"],
                            control=control_input[..., :width],
                        ).sample.to(dtype=latents_dtype)
                        values[..., :latent_width] += noise_pred_left
                        counts[..., :latent_width] += 1
                        noise_pred_right = self.unet(
                            latent_model_input[..., -latent_width:],
                            t,
                            encoder_hidden_states=text_embeddings["embedding_r"],
                            control=control_input[..., -width:],
                        ).sample.to(dtype=latents_dtype)
                        values[..., -latent_width:] += noise_pred_right
                        counts[..., -latent_width:] += 1

                        noise_pred_top = self.unet(
                            latent_model_input[..., :latent_height, :],
                            t,
                            encoder_hidden_states=text_embeddings["embedding_t"],
                            control=control_input[..., :height, :, :],
                        ).sample.to(dtype=latents_dtype)
                        values[..., :latent_height, :] += noise_pred_top
                        counts[..., :latent_height, :] += 1
                        noise_pred_bottom = self.unet(
                            latent_model_input[..., -latent_height:, :],
                            t,
                            encoder_hidden_states=text_embeddings["embedding_b"],
                            control=control_input[..., -height:, :],
                        ).sample.to(dtype=latents_dtype)
                        values[..., -latent_height:, :] += noise_pred_bottom
                        counts[..., -latent_height:, :] += 1

                        noise_pred_middle = self.unet(
                            latent_model_input[
                                ...,
                                latent_expand_t : latent_expand_t + latent_height,
                                latent_expand_l : latent_expand_l + latent_width,
                            ],
                            t,
                            encoder_hidden_states=text_embeddings,
                            control=control_input[
                                ...,
                                expand_t : expand_t + height,
                                expand_l : expand_l + width,
                            ],
                        ).sample.to(dtype=latents_dtype)
                        values[
                            ...,
                            latent_expand_t : latent_expand_t + latent_height,
                            latent_expand_l : latent_expand_l + width,
                        ] += noise_pred_middle
                        counts[
                            ...,
                            latent_expand_t : latent_expand_t + latent_height,
                            latent_expand_l : latent_expand_l + width,
                        ] += 1
                        noise_pred = values / counts
                    else:
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings["text_embedding"],
                            control=control_input,
                        ).sample.to(dtype=latents_dtype)
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    if (
                        i > int(warp_step[0] * len(timesteps))
                        and i < int(warp_step[1] * len(timesteps))
                        and k < warp_time
                        and bwd_mask is not None
                    ):
                        # assert 0,"should not be here right now"
                        extra_step_kwargs["latent_expand_l"] = latent_expand_l
                        extra_step_kwargs["latent_expand_r"] = latent_expand_r
                        extra_step_kwargs["latent_expand_t"] = latent_expand_t
                        extra_step_kwargs["latent_expand_b"] = latent_expand_b
                        extra_step_kwargs["latent_height"] = latent_height
                        extra_step_kwargs["latent_width"] = latent_width
                        values_tot[:, :, seq] += self.scheduler.step(
                            noise_pred,
                            t,
                            latents[:, :, seq],
                            gt=init_latents,
                            **flows,
                            **extra_step_kwargs,
                        ).prev_sample

                    else:
                        values_tot[:, :, seq] += self.scheduler.step(
                            noise_pred, t, latents[:, :, seq], **extra_step_kwargs
                        ).prev_sample
                    counts_tot[:, :, seq] += 1

                latents = values_tot / counts_tot

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents,
                        noise[
                            ...,
                            latent_expand_t : latent_expand_t + latent_height,
                            latent_expand_l : latent_expand_l + latent_width,
                        ],
                        torch.tensor([noise_timestep]),
                    )

                    latents[
                        ...,
                        latent_expand_t : latent_expand_t + latent_height,
                        latent_expand_l : latent_expand_l + latent_width,
                    ] = init_latents_proper

                i += 1
                if (
                    jump_length != 0
                    and i % jump_length == 0
                    and i > 0
                    and i < int(0.6 * len(timesteps))
                ):
                    if k < repeat_time:
                        current_timestep = timesteps[i]
                        target_timestep = timesteps[i - jump_length]
                        noise = torch.randn_like(latents)
                        latents = self.scheduler.noise_travel(
                            latents,
                            noise,
                            torch.tensor([current_timestep]),
                            torch.tensor([target_timestep]),
                        )
                        i = i - jump_length
                        k += 1
                    else:
                        k = 0
                        for _ in range(jump_length):
                            progress_bar.update()
                else:
                    progress_bar.update()

            for _ in range(jump_length - 1):
                progress_bar.update()

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return OutpaintPipelineOutput(videos=video)


def get_views(video_length, window_size=16, stride=4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append(list(range(t_start,t_end)))
    return views