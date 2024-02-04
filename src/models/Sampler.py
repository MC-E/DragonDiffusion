
from diffusers import StableDiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn.functional as F
import torch
from basicsr.utils import img2tensor
from tqdm import tqdm
import torch.nn as nn
import copy
import numpy as np

class Sampler(StableDiffusionPipeline):

    def edit(
        self,
        prompt:  List[str],
        mode,
        emb_im,
        emb_im_uncond,
        edit_kwargs,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        latent: Optional[torch.FloatTensor] = None,
        start_time=50,
        energy_scale = 0,
        SDE_strength = 0.4,
        SDE_strength_un = 0,
        latent_noise_ref = None,
        alg='D+'
    ):
        print('Start Editing:')
        self.alg=alg
        # generate source text embedding
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        # image prompt
        if emb_im is not None and emb_im_uncond is not None:
            uncond_embeddings = torch.cat([uncond_embeddings, emb_im_uncond],dim=1)
            text_embeddings_org = text_embeddings
            text_embeddings = torch.cat([text_embeddings, emb_im],dim=1)
            context = torch.cat([uncond_embeddings.expand(*text_embeddings.shape), text_embeddings])

        self.scheduler.set_timesteps(num_inference_steps) 
        dict_mask = edit_kwargs['dict_mask'] if 'dict_mask' in edit_kwargs else None

        for i, t in enumerate(tqdm(self.scheduler.timesteps[-start_time:])):
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            if energy_scale==0 or alg=='D':
                repeat=1
            elif 20<i<30 and i%2==0 : 
                repeat = 3
            else:
                repeat = 1
            stack = []
            for ri in range(repeat):
                latent_in = torch.cat([latent.unsqueeze(2)] * 2)
                with torch.no_grad():
                    noise_pred = self.unet(latent_in, t, encoder_hidden_states=context, mask=dict_mask, save_kv=False, mode=mode, iter_cur=i)["sample"].squeeze(2)
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

                if energy_scale!=0 and i<30 and (alg=='D' or i%2==0 or i<10):
                    # editing guidance
                    noise_pred_org = noise_pred
                    if mode == 'move':
                        guidance = self.guidance_move(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'drag':
                        guidance = self.guidance_drag(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'landmark':
                        guidance = self.guidance_landmark(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'appearance':
                        guidance = self.guidance_appearance(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'paste':
                        guidance = self.guidance_paste(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)

                    noise_pred = noise_pred + guidance
                else:
                    noise_pred_org=None
                # zt->zt-1
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latent - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                if 10<i<20:
                    eta, eta_rd = SDE_strength_un, SDE_strength
                else:
                    eta, eta_rd = 0., 0.
                
                variance = self.scheduler._get_variance(t, prev_timestep)
                std_dev_t = eta * variance ** (0.5)
                std_dev_t_rd = eta_rd * variance ** (0.5)
                if noise_pred_org is not None:
                    pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred_org
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred_org
                else:
                    pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

                latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd

                # Regional SDE
                if (eta_rd > 0 or eta>0) and alg=='D+':
                    variance_noise = torch.randn_like(latent_prev)
                    variance_rd = std_dev_t_rd * variance_noise
                    variance = std_dev_t * variance_noise
                    
                    if mode == 'move':
                        mask = (F.interpolate(edit_kwargs["mask_x0"][None,None], (edit_kwargs["mask_cur"].shape[-2], edit_kwargs["mask_cur"].shape[-1]))>0.5).float()
                        mask = ((edit_kwargs["mask_cur"]+mask)>0.5).float()
                        mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latent.dtype)
                    elif mode == 'drag':
                        mask = F.interpolate(edit_kwargs["mask_x0"][None,None], (latent_prev[-1].shape[-2], latent_prev[-1].shape[-1]))
                        mask = (mask>0).to(dtype=latent.dtype)
                    elif mode == 'landmark':
                        mask = torch.ones_like(latent_prev)
                    elif mode == 'appearance' or mode == 'paste':
                        mask = F.interpolate(edit_kwargs["mask_base_cur"].float(), (latent_prev[-1].shape[-2], latent_prev[-1].shape[-1]))
                        mask = (mask>0).to(dtype=latent.dtype)
                    latent_prev = (latent_prev+variance)*(1-mask) + (latent_prev_rd+variance_rd)*mask

                if repeat>1:
                    with torch.no_grad():
                        alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
                        alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        model_output = self.unet(latent_prev.unsqueeze(2), next_timestep, encoder_hidden_states=text_embeddings, mask=dict_mask, save_kv=False, mode=mode, iter_cur=-2)["sample"].squeeze(2)
                        next_original_sample = (latent_prev - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
                        latent = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
            
            latent = latent_prev
            
        return latent

    def guidance_move(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
    ):
        cos = nn.CosineSimilarity(dim=1)
        loss_scale = [0.5, 0.5]
        with torch.no_grad():
            up_ft_tar = self.estimator(
                        sample=latent_noise_ref.squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))

        latent = latent.detach().requires_grad_(True)
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*resize_scale*up_scale), int(up_ft_tar[-1].shape[-1]*resize_scale*up_scale)))

        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        # editing energy
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id] 

        # content energy
        loss_con = 0
        if mask_x0_ref is not None:
            mask_x0_ref_cur = F.interpolate(mask_x0_ref[None,None], (mask_other.shape[-2], mask_other.shape[-1]))>0.5
        else:
            mask_x0_ref_cur = mask_other
        for f_id in range(len(up_ft_tar_org)):
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]

        for f_id in range(len(up_ft_tar)):
            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_non_overlap.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1], -1).permute(1,0)
            sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
            loss_con = loss_con + w_contrast*sim_non_overlap.mean()*loss_scale[f_id]

            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_x0_ref_cur.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            sim_inpaint = ((cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.)
            loss_con = loss_con + w_inpaint/(1+4*sim_inpaint.mean())

        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        mask_edit2 = (F.interpolate(mask_x0[None,None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float()
        mask_edit1 = (mask_cur>0.5).float()
        mask = ((mask_cur+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        self.estimator.zero_grad()

        return guidance

    def guidance_drag(
        self, 
        mask_x0,
        mask_cur, 
        mask_tar, 
        mask_other, 
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        up_scale, 
        text_embeddings,
        energy_scale,
        w_edit,
        w_inpaint,
        w_content,
        dict_mask = None,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar = self.estimator(
                        sample=latent_noise_ref.squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar)):
                up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (up_ft_tar[-1].shape[-2]*up_scale, up_ft_tar[-1].shape[-1]*up_scale))

        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))

        # moving loss
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                up_ft_cur_vec = up_ft_cur[f_id][mask_cur_i.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_tar_vec = up_ft_tar[f_id][mask_tar_i.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
                sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
                loss_edit = loss_edit + w_edit/(1+4*sim.mean())

                mask_overlap = ((mask_cur_i.float()+mask_tar_i.float())>1.5).float()
                mask_non_overlap = (mask_tar_i.float()-mask_overlap)>0.5
                up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_tar_non_overlap = up_ft_tar[f_id][mask_non_overlap.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
                sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
                loss_edit = loss_edit + w_inpaint*sim_non_overlap.mean()
        # consistency loss
        loss_con = 0
        for f_id in range(len(up_ft_tar)):
            sim_other = (cos(up_ft_tar[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]+1.)/2.
            loss_con = loss_con+w_content/(1+4*sim_other.mean())
        loss_edit = loss_edit/len(up_ft_cur)/len(mask_cur)
        loss_con = loss_con/len(up_ft_cur)

        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        mask = F.interpolate(mask_x0[None,None], (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        guidance = cond_grad_edit.detach()*4e-2*mask + cond_grad_con.detach()*4e-2*(1-mask)
        self.estimator.zero_grad()

        return guidance

    def guidance_landmark(
        self, 
        mask_cur, 
        mask_tar,
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        up_scale, 
        text_embeddings,
        energy_scale,
        w_edit,
        w_inpaint,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar = self.estimator(
                        sample=latent_noise_ref.squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar)):
                up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (up_ft_tar[-1].shape[-2]*up_scale, up_ft_tar[-1].shape[-1]*up_scale))

        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))

        # moving loss
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                up_ft_cur_vec = up_ft_cur[f_id][mask_cur_i.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_tar_vec = up_ft_tar[f_id][mask_tar_i.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
                sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
                loss_edit = loss_edit + w_edit/(1+4*sim.mean())
        loss_edit = loss_edit/len(up_ft_cur)/len(mask_cur)

        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        guidance = cond_grad_edit.detach()*4e-2
        self.estimator.zero_grad()

        return guidance

    def guidance_appearance(
        self, 
        mask_base_cur, 
        mask_replace_cur, 
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        up_scale, 
        text_embeddings,
        energy_scale,
        dict_mask,
        w_edit,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar_base = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(up_ft_tar_base[f_id], (up_ft_tar_base[-1].shape[-2]*up_scale, up_ft_tar_base[-1].shape[-1]*up_scale))
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[1::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        
        # for base content
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1-mask_base_cur.float())>0.5
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar_base[f_id][mask_cur.repeat(1,up_ft_tar_base[f_id].shape[1],1,1)].view(up_ft_tar_base[f_id].shape[1], -1).permute(1,0)
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
            loss_con = loss_con + w_content/(1+4*sim.mean())
        # for replace content
        loss_edit = 0
        for f_id in range(len(up_ft_tar_replace)):
            mask_cur = mask_base_cur
            mask_tar = mask_replace_cur
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0).mean(0, keepdim=True)
            up_ft_tar_vec = up_ft_tar_replace[f_id][mask_tar.repeat(1,up_ft_tar_replace[f_id].shape[1],1,1)].view(up_ft_tar_replace[f_id].shape[1], -1).permute(1,0).mean(0, keepdim=True)
            sim_all=((cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.)
            loss_edit =  loss_edit + w_edit/(1+4*sim_all.mean())

        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent, retain_graph=True)[0]
        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent)[0]
        mask = F.interpolate(mask_base_cur.float(), (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        guidance = cond_grad_con.detach()*(1-mask)*4e-2 + cond_grad_edit.detach()*mask*4e-2
        self.estimator.zero_grad()

        return guidance

    def guidance_paste(
        self, 
        mask_base_cur, 
        mask_replace_cur, 
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        up_scale, 
        text_embeddings,
        energy_scale,
        dict_mask,
        w_edit,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar_base = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(up_ft_tar_base[f_id], (up_ft_tar_base[-1].shape[-2]*up_scale, up_ft_tar_base[-1].shape[-1]*up_scale))
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[1::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
            
        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        # for base content
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1-mask_base_cur.float())>0.5
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar_base[f_id][mask_cur.repeat(1,up_ft_tar_base[f_id].shape[1],1,1)].view(up_ft_tar_base[f_id].shape[1], -1).permute(1,0)
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
            loss_con = loss_con + w_content/(1+4*sim.mean())
        # for replace content
        loss_edit = 0
        for f_id in range(len(up_ft_tar_replace)):
            mask_cur = mask_base_cur

            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar_replace[f_id][mask_replace_cur.repeat(1,up_ft_tar_replace[f_id].shape[1],1,1)].view(up_ft_tar_replace[f_id].shape[1], -1).permute(1,0)
            sim_all=((cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.)
            loss_edit =  loss_edit + w_edit/(1+4*sim_all.mean())

        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent, retain_graph=True)[0]
        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent)[0]
        mask = F.interpolate(mask_base_cur.float(), (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        guidance = cond_grad_con.detach()*(1-mask)*4e-2 + cond_grad_edit.detach()*mask*4e-2
        self.estimator.zero_grad()

        return guidance