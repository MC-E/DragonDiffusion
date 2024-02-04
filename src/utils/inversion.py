from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
import numpy as np

class DDIMInversion:
    def __init__(self, model, NUM_DDIM_STEPS):
        self.model = model
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.prompt = None
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context, iter_cur):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context, iter_cur=iter_cur)["sample"]
        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt: str, emb_im=None):
        if not isinstance(prompt, list):
            prompt = [prompt]
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        if emb_im is not None:
            self.text_embeddings = torch.cat([text_embeddings, emb_im],dim=1)
        else:
            self.text_embeddings = text_embeddings

        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        cond_embeddings = self.text_embeddings
        all_latent = [latent]
        latent = latent.clone().detach()
        print('DDIM Inversion:')
        for i in tqdm(range(self.NUM_DDIM_STEPS)):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings, iter_cur=len(self.model.scheduler.timesteps) - i - 1)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler
    
    def invert(self, ddim_latents, prompt: str, emb_im=None):
        self.init_prompt(prompt, emb_im=emb_im)
        ddim_latents = self.ddim_loop(ddim_latents)
        return ddim_latents