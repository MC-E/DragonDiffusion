
import torch
import torch.nn as nn
import numpy as np
from src.unet.unet_2d_condition import DragonUNet2DConditionModel
from src.unet.estimator import MyUNet2DConditionModel
from diffusers import DDIMScheduler
import gc
from PIL import Image
from basicsr.utils import tensor2img
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from src.utils.inversion import DDIMInversion
from src.unet.attention_processor import IPAttnProcessor, AttnProcessor, Resampler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.models.Sampler import Sampler

class DragonPipeline:
    def __init__(self, sd_id='stabilityai/stable-diffusion-2-1', ip_id='models/ip_sd15_64.bin', NUM_DDIM_STEPS=50, precision=torch.float32, ip_scale=0):
        unet = DragonUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", torch_dtype=precision)
        tokenizer = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder", torch_dtype=precision)
        onestep_pipe = Sampler.from_pretrained(sd_id, unet=unet, safety_checker=None, feature_extractor=None, tokenizer=tokenizer, text_encoder=text_encoder, dtype=precision)
        onestep_pipe.vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae", torch_dtype=precision)
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        onestep_pipe.estimator = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet",vae=None, text_encoder=None, tokenizer=None,
                                scheduler=DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler"),
                                safety_checker=None, feature_extractor=None,).to('cuda', dtype=precision)
        onestep_pipe.estimator.enable_xformers_memory_efficient_attention()
        gc.collect()
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.precision = precision
        self.ip_id = ip_id
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained('h94/IP-Adapter', subfolder='models/image_encoder').to('cuda', dtype=precision)
        self.clip_image_processor = CLIPImageProcessor()
        self.num_tokens=64
        self.image_proj_model = self.init_proj(precision)
        self.load_adapter(ip_id, ip_scale)

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        img = self.pipe.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        img = tensor2img(img.cpu().float())
        if isinstance(img, list):
            img = img[0] 
        return img

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            latents = self.pipe.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
        return latents

    def ddim_inv(self, latent, prompt, emb_im=None):
        ddim_inv = DDIMInversion(model=self.pipe, NUM_DDIM_STEPS=self.NUM_DDIM_STEPS)
        ddim_latents = ddim_inv.invert(ddim_latents=latent.unsqueeze(2), prompt=prompt, emb_im=emb_im)
        return ddim_latents

    def init_proj(self, precision):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4
        ).to('cuda', dtype=precision)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to('cuda', dtype=self.precision)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2].detach()
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds).detach()
        return image_prompt_embeds, uncond_image_prompt_embeds

    def load_adapter(self, model_path, scale=1.0):
        attn_procs = {}
        for name in self.pipe.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.pipe.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.pipe.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.pipe.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.pipe.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                scale=scale,num_tokens= self.num_tokens).to('cuda', dtype=self.precision)
        self.pipe.unet.set_attn_processor(attn_procs)
        state_dict = torch.load(model_path, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=True)