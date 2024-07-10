import argparse
import copy
import logging
import math
import os
from typing import Any, Dict, Optional, Tuple
from omegaconf import OmegaConf 
import torch
import torch.utils.checkpoint

import diffusers
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL

from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.schedulers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from src.data.naive_dataset import TuneAVideoDataset
from src.data.naive_dataset import TuneAVideoDatasetSplit

from src.models.pretrain.unet import UNet3DConditionModel
from src.pipelines.pipelineoutpaint import OutpaintPipeline
from src.util import ddim_inversion, save_videos_grid
from src.schedulers.scheduling_ddim import DDIMScheduler
from src.models.vanilla.controlnet import ControlNetModel
from einops import rearrange, repeat
from diffusers.optimization import get_scheduler
import torch.nn.functional as F
import numpy as np

logger = get_logger(__name__, log_level="INFO")


def count_parameters(module):
    """
    Counts the total number of parameters in a PyTorch nn module and 
    returns the size in megabytes (MB).

    Args:
    module (nn.Module): The PyTorch module to count parameters for.

    Returns:
    float: The total size of parameters in MB.
    """
    total_params = sum(p.numel() for p in module.parameters())
    total_size_bytes = total_params * 4  # Assuming 32-bit floats (4 bytes)
    total_size = total_size_bytes / (1024 ** 2)  # Convert bytes to MB
    return total_size


class MaskGenerator():
    def __init__(self, mask_l, mask_r, mask_t, mask_b) -> None:
        self.mask_l = mask_l
        self.mask_r = mask_r
        self.mask_t = mask_t
        self.mask_b = mask_b
        
    
    def __call__(self, control):
        mask = - torch.ones_like(control)
        b, c, f, h, w = mask.shape
        
        l = np.random.rand() * (self.mask_l[1] - self.mask_l[0])   + self.mask_l[0]
        r = np.random.rand() * (self.mask_r[1] - self.mask_r[0])   + self.mask_r[0]
        t = np.random.rand() * (self.mask_t[1] - self.mask_t[0])   + self.mask_t[0]
        b = np.random.rand() * (self.mask_b[1] - self.mask_b[0])   + self.mask_b[0]
        l, r, t, b = int(l*w), int(r*w), int(t*h), int(b*h)
        
        if r == 0 and b==0:
            mask[...,t:,l:] = control[...,t:,l:]
        elif b == 0:
            mask[...,t:,l:-r] = control[...,t:,l:-r] 
        elif r == 0:
            mask[...,t:-b,l:] = control[...,t:-b,l:]
        else:
            mask[...,t:-b,l:-r] = control[...,t:-b,l:-r]
                    
        return mask
        

def main(base_config, config):

    accelerator = Accelerator(
        gradient_accumulation_steps=base_config.training_config.gradient_accumulation_steps,
        mixed_precision=base_config.training_config.mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if base_config.training_config.seed is not None:
        set_seed(base_config.training_config.seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(base_config.output_dir, exist_ok=True)
        os.makedirs(f"{base_config.output_dir}/samples", exist_ok=True)
        # os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(base_config, os.path.join(base_config.output_dir, 'base_config.yaml'))
        OmegaConf.save(config,os.path.join(base_config.output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler(**OmegaConf.to_container(base_config.noise_scheduler_kwargs))
    tokenizer = CLIPTokenizer.from_pretrained(base_config.model_config.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_config.model_config.sd_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_config.model_config.sd_path, subfolder="vae")
    # vae = AutoencoderKL.from_pretrained(base_config.model_config.vae_path)
    unet = UNet3DConditionModel.from_pretrained_2d(base_config.model_config.sd_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet = ControlNetModel.from_pretrained_2d(base_config.model_config.control_path)
    motion_module_state_dict = torch.load(base_config.model_config.temporal_path, map_location="cpu")
    missing, unexpected = unet.load_state_dict(motion_module_state_dict, strict=False)
    unet.controlnet = controlnet
    unet.controlnet_scale = base_config.model_config.control_scale

    
    trainable_params = []
    unet.requires_grad_(False)
    if controlnet is not None:
        controlnet.requires_grad_(False)

            
    for name, module in unet.named_modules():
        if base_config.training_config.trainable_modules is not None and name.endswith(tuple(base_config.training_config.trainable_modules)) and "controlnet"  not in name:
            print(name)
            # assert "to_q" in name
            tmp = []
            for params in module.parameters():
                params.requires_grad = True
                tmp.append(params)
            dct = {"params":tmp,"lr":base_config.training_config.learning_rate}
            trainable_params.append(dct)
                # trainable_params.append(params)
                
    if "lora" in base_config.training_config.trainable_modules:
        from src.utils.register import register_motion_lora, adjust_motion_lora
        print(count_parameters(unet)+count_parameters(vae)+count_parameters(text_encoder))
        loras = register_motion_lora(unet)
        print(count_parameters(loras))
        tmp = []
        for param in loras.parameters():
            param.requires_grad = True
            tmp.append(param)
        dct = {"params":tmp,"lr":1e-4}
        trainable_params.append(dct)
    elif "lora_moe" in base_config.training_config.trainable_modules:
        from src.utils.register_moe import register_motion_lora, adjust_motion_lora
        loras = register_motion_lora(unet)
        for param in loras.parameters():
            param.requires_grad = True
            trainable_params.append(param)
    elif "sv" in base_config.training_config.trainable_modules:
        from src.utils.register_svdiff import register_motion_lora, adjust_motion_lora
        loras = register_motion_lora(unet)
        tmp = []
        for param in loras.parameters():
            param.requires_grad = True
            tmp.append(param)
        dct = {"params":tmp,"lr":5e-3}
        trainable_params.append(dct)
    else:
        loras = None
 
 
    if base_config.training_config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if base_config.training_config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if base_config.training_config.scale_lr:
        learning_rate = (
            learning_rate * base_config.training_config.gradient_accumulation_steps * base_config.training_config.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if base_config.training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params,
        lr=base_config.training_config.learning_rate,
        betas=(base_config.training_config.adam_beta1, base_config.training_config.adam_beta2),
        weight_decay=base_config.training_config.adam_weight_decay,
        eps=base_config.training_config.adam_epsilon,
    )
    trainable_params = [param for dct in trainable_params for param in dct['params']]

    train_dataset = TuneAVideoDatasetSplit(**config.train_data)

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=base_config.training_config.train_batch_size
    )

    # Get the validation pipeline
    validation_pipeline = OutpaintPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(base_config.noise_scheduler_kwargs))
    )
    validation_pipeline.enable_vae_slicing()

    # Scheduler
    lr_scheduler = get_scheduler(
        base_config.training_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=base_config.training_config.lr_warmup_steps * base_config.training_config.gradient_accumulation_steps,
        num_training_steps=base_config.training_config.max_train_steps * base_config.training_config.gradient_accumulation_steps,
    )

    if loras is not None:
        unet, optimizer, train_dataloader, lr_scheduler, loras = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, loras
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / base_config.training_config.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(base_config.training_config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = base_config.training_config.train_batch_size * accelerator.num_processes * base_config.training_config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {base_config.training_config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {base_config.training_config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {base_config.training_config.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step,base_config.training_config.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    
    mask_generator = MaskGenerator(**config.mask_config)
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step

            unet.controlnet_scale = 1.
            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                latents = validation_pipeline.encode_videos(pixel_values)
                    
                if global_step == 0:
                    # Validation for none inpainting block is used
                    full_video = batch["full_video"][0:1].to(weight_dtype)
                    video_length = full_video.shape[1]
                    full_video = rearrange(full_video, "b f c h w -> b c f h w")
                    full_latents = validation_pipeline.encode_videos(full_video)
                    config.validation_data.video_length = video_length
                    
                    unet.controlnet_scale = 0
                    samples = []
                    for idx, (prompt,prompt_l,prompt_r,prompt_t,prompt_b,prompt_neg) in enumerate(zip(config.validation_data.prompts,config.validation_data.prompts_l,config.validation_data.prompts_r,config.validation_data.prompts_t,config.validation_data.prompts_b,config.validation_data.prompts_neg)):
                        with torch.autocast("cuda"):
                            sample_wide = validation_pipeline(
                                prompt = prompt,
                                prompt_l = prompt_l,
                                prompt_r = prompt_r,
                                prompt_t = prompt_t,
                                prompt_b = prompt_b,
                                negative_prompt = prompt_neg,
                                init_video = full_video,
                                **config.validation_data
                                ).videos  
                        samples.append(sample_wide)
                    samples = torch.concat(samples)
                    save_path = f"{base_config.output_dir}/samples/sample-baseline-wocontrol-long.gif"
                    save_videos_grid(samples, save_path)
                    logger.info(f"Saved samples to {save_path}")  
                    
                    unet.controlnet_scale = base_config.model_config.control_scale
                    # Validation for basic use of inpainting block and temporal block
                    samples = []
                    for idx, (prompt,prompt_l,prompt_r,prompt_t,prompt_b,prompt_neg) in enumerate(zip(config.validation_data.prompts,config.validation_data.prompts_l,config.validation_data.prompts_r,config.validation_data.prompts_t,config.validation_data.prompts_b,config.validation_data.prompts_neg)):
                        with torch.autocast("cuda"):
                            sample_wide = validation_pipeline(
                                prompt = prompt,
                                prompt_l = prompt_l,
                                prompt_r = prompt_r,
                                prompt_t = prompt_t,
                                prompt_b = prompt_b,
                                negative_prompt = prompt_neg,
                                init_video = full_video,
                                **config.validation_data
                                ).videos  
                        samples.append(sample_wide)
                    samples = torch.concat(samples)
                    save_path = f"{base_config.output_dir}/samples/sample-baseline-long.gif"
                    save_videos_grid(samples, save_path)
                    logger.info(f"Saved samples to {save_path}")  


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                control = (pixel_values+1)/2.

                control = mask_generator(control)
                
                mask = (control == -1).float()
                mask = F.interpolate(mask,scale_factor=(1,1/8,1/8))[:,0:1]
                
                
                
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, control=control).sample
                mask = torch.ones_like(target)
                loss = (F.mse_loss(model_pred.float(),target.float(),reduction="none") * mask).mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(base_config.training_config.train_batch_size)).mean()
                train_loss += avg_loss.item() / base_config.training_config.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, base_config.training_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % base_config.training_config.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(base_config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % base_config.training_config.validation_steps == 0:
                    if accelerator.is_main_process:
                        
                        
                        if loras is not None:
                            loras = adjust_motion_lora(unet,loras,config.validation_data.scale_l, config.validation_data.scale_r, config.validation_data.scale_t, config.validation_data.scale_b,config.validation_data.height,config.validation_data.width,k=1)
                                                                 
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(base_config.training_config.seed)
                        unet.controlnet_scale = 1
                        samples = []
                        for idx, (prompt,prompt_l,prompt_r,prompt_t,prompt_b,prompt_neg) in enumerate(zip(config.validation_data.prompts,config.validation_data.prompts_l,config.validation_data.prompts_r,config.validation_data.prompts_t,config.validation_data.prompts_b,config.validation_data.prompts_neg)):
                            with torch.autocast("cuda"):
                                sample_wide = validation_pipeline(
                                    prompt = prompt,
                                    prompt_l = prompt_l,
                                    prompt_r = prompt_r,
                                    prompt_t = prompt_t,
                                    prompt_b = prompt_b,
                                    negative_prompt = prompt_neg,
                                    init_video = full_video,
                                    **config.validation_data
                                    ).videos  
                            samples.append(sample_wide)
                        samples = torch.concat(samples)
                        save_path = f"{base_config.output_dir}/samples/sample-{global_step}-long.gif"
                        save_videos_grid(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")  
                        if loras is not None:
                            loras = adjust_motion_lora(unet,loras,config.validation_data.scale_l, config.validation_data.scale_r, config.validation_data.scale_t, config.validation_data.scale_b,config.validation_data.height,config.validation_data.width,inference=False)
        
        
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= base_config.training_config.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default="./configs/base.yaml")
    parser.add_argument("--config",type=str,default="./configs/exp.yaml")
    args = parser.parse_args()
    
    base_configs = OmegaConf.load(args.base_config)
    configs = OmegaConf.load(args.config)
    
    
    
    for base_key, base_config in base_configs.items():
        base_output_dir = copy.deepcopy(base_config.output_dir)
        for key, config in list(configs.items()):
            base_config.output_dir = os.path.join(os.path.join(base_output_dir,base_key),str(key)) 
            try:
                main(base_config, config)
            except Exception as e:
                print(f"An exception occurred: {e}")
                print("fail at",key)
                