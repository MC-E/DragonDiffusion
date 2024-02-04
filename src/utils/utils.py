import numpy as np
import cv2
from basicsr.utils import img2tensor
import torch
import torch.nn.functional as F

def resize_numpy_image(image, max_resolution=768 * 768, resize_short_edge=None):
    h, w = image.shape[:2]
    w_org = image.shape[1]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    scale = w/w_org
    return image, scale

def split_ldm(ldm):
    x = []
    y = []
    for p in ldm:
        x.append(p[0])
        y.append(p[1])
    return x,y

def process_move(path_mask, h, w, dx, dy, scale, input_scale, resize_scale, up_scale, up_ft_index, w_edit, w_content, w_contrast, w_inpaint,  precision, path_mask_ref=None):
    dx, dy = dx*input_scale, dy*input_scale
    if isinstance(path_mask, str):
        mask_x0 = cv2.imread(path_mask)
    else:
        mask_x0 = path_mask
    mask_x0 = cv2.resize(mask_x0, (h, w))
    if path_mask_ref is not None:
        if isinstance(path_mask_ref, str):
            mask_x0_ref = cv2.imread(path_mask_ref)
        else:
            mask_x0_ref = path_mask_ref
        mask_x0_ref = cv2.resize(mask_x0_ref, (h, w))
    else:
        mask_x0_ref=None

    mask_x0 = img2tensor(mask_x0)[0]
    mask_x0 = (mask_x0>0.5).float().to('cuda', dtype=precision)
    if mask_x0_ref is not None:
        mask_x0_ref = img2tensor(mask_x0_ref)[0]
        mask_x0_ref = (mask_x0_ref>0.5).float().to('cuda', dtype=precision)
    mask_org = F.interpolate(mask_x0[None,None], (int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)))>0.5

    mask_tar = F.interpolate(mask_x0[None,None], (int(mask_x0.shape[-2]//scale*resize_scale), int(mask_x0.shape[-1]//scale*resize_scale)))>0.5
    mask_cur = torch.roll(mask_tar, (int(dy//scale*resize_scale), int(dx//scale*resize_scale)), (-2,-1))
    
    pad_size_x = abs(mask_tar.shape[-1]-mask_org.shape[-1])//2
    pad_size_y = abs(mask_tar.shape[-2]-mask_org.shape[-2])//2
    if resize_scale>1:
        sum_before = torch.sum(mask_cur)
        mask_cur = mask_cur[:,:,pad_size_y:pad_size_y+mask_org.shape[-2],pad_size_x:pad_size_x+mask_org.shape[-1]]
        sum_after = torch.sum(mask_cur)
        if sum_after != sum_before:
            raise ValueError('Resize out of bounds, exiting.')
    else:
        temp = torch.zeros(1,1,mask_org.shape[-2], mask_org.shape[-1]).to(mask_org.device)
        temp[:,:,pad_size_y:pad_size_y+mask_cur.shape[-2],pad_size_x:pad_size_x+mask_cur.shape[-1]]=mask_cur
        mask_cur =temp>0.5

    mask_other = (1-((mask_cur+mask_org)>0.5).float())>0.5
    mask_overlap = ((mask_cur.float()+mask_org.float())>1.5).float()
    mask_non_overlap = (mask_org.float()-mask_overlap)>0.5

    return {
        "mask_x0":mask_x0, 
        "mask_x0_ref":mask_x0_ref, 
        "mask_tar":mask_tar, 
        "mask_cur":mask_cur, 
        "mask_other":mask_other, 
        "mask_overlap":mask_overlap, 
        "mask_non_overlap":mask_non_overlap, 
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "resize_scale":resize_scale,
        "w_edit":w_edit,
        "w_content":w_content,
        "w_contrast":w_contrast,
        "w_inpaint":w_inpaint, 
    }

def process_drag_face(h, w, x, y, x_cur, y_cur, scale, input_scale, up_scale, up_ft_index, w_edit, w_inpaint, precision):
    for i in range(len(x)):
        x[i] = int(x[i]*input_scale)
        y[i] = int(y[i]*input_scale)
        x_cur[i] = int(x_cur[i]*input_scale)
        y_cur[i] = int(y_cur[i]*input_scale)

    mask_tar = []
    for p_idx in range(len(x)):
        mask_i = torch.zeros(int(h//scale), int(w//scale)).cuda()
        y_clip = int(np.clip(y[p_idx]//scale, 1, mask_i.shape[0]-2))
        x_clip = int(np.clip(x[p_idx]//scale, 1, mask_i.shape[1]-2))
        mask_i[y_clip-1:y_clip+2,x_clip-1:x_clip+2]=1
        mask_i = mask_i>0.5
        mask_tar.append(mask_i)
    mask_cur = []
    for p_idx in range(len(x_cur)):
        mask_i = torch.zeros(int(h//scale), int(w//scale)).cuda()
        y_clip = int(np.clip(y_cur[p_idx]//scale, 1, mask_i.shape[0]-2))
        x_clip = int(np.clip(x_cur[p_idx]//scale, 1, mask_i.shape[1]-2))
        mask_i[y_clip-1:y_clip+2,x_clip-1:x_clip+2]=1
        mask_i=mask_i>0.5
        mask_cur.append(mask_i)

    return {
        "mask_tar":mask_tar,
        "mask_cur":mask_cur,
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "w_edit": w_edit,
        "w_inpaint": w_inpaint,
    }

def process_drag(path_mask, h, w, x, y, x_cur, y_cur, scale, input_scale, up_scale, up_ft_index, w_edit, w_inpaint, w_content, precision, latent_in):
    if isinstance(path_mask, str):
        mask_x0 = cv2.imread(path_mask)
    else:
        mask_x0 = path_mask
    mask_x0 = cv2.resize(mask_x0, (h, w))
    mask_x0 = img2tensor(mask_x0)[0]
    dict_mask = {}
    dict_mask['base'] = mask_x0
    mask_x0 = (mask_x0>0.5).float().to('cuda', dtype=precision)

    mask_other = F.interpolate(mask_x0[None,None], (int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)))<0.5
    mask_tar = []
    mask_cur = []
    for p_idx in range(len(x)):
        mask_tar_i = torch.zeros(int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)).to('cuda', dtype=precision)
        mask_cur_i = torch.zeros(int(mask_x0.shape[-2]//scale), int(mask_x0.shape[-1]//scale)).to('cuda', dtype=precision)
        y_tar_clip = int(np.clip(y[p_idx]//scale, 1, mask_tar_i.shape[0]-2))
        x_tar_clip = int(np.clip(x[p_idx]//scale, 1, mask_tar_i.shape[0]-2))
        y_cur_clip = int(np.clip(y_cur[p_idx]//scale, 1, mask_cur_i.shape[0]-2))
        x_cur_clip = int(np.clip(x_cur[p_idx]//scale, 1, mask_cur_i.shape[0]-2))
        mask_tar_i[y_tar_clip-1:y_tar_clip+2,x_tar_clip-1:x_tar_clip+2]=1
        mask_cur_i[y_cur_clip-1:y_cur_clip+2,x_cur_clip-1:x_cur_clip+2]=1
        mask_tar_i = mask_tar_i>0.5
        mask_cur_i=mask_cur_i>0.5
        mask_tar.append(mask_tar_i)
        mask_cur.append(mask_cur_i)
        latent_in[:,:,y_cur_clip//up_scale-1:y_cur_clip//up_scale+2, x_cur_clip//up_scale-1:x_cur_clip//up_scale+2] = latent_in[:,:, y_tar_clip//up_scale-1:y_tar_clip//up_scale+2, x_tar_clip//up_scale-1:x_tar_clip//up_scale+2] 
        

    return {
        "dict_mask":dict_mask,
        "mask_x0":mask_x0,
        "mask_tar":mask_tar,
        "mask_cur":mask_cur,
        "mask_other":mask_other,
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "w_edit": w_edit,
        "w_inpaint": w_inpaint,
        "w_content": w_content,
        "latent_in":latent_in,
    }

def process_appearance(path_mask, path_mask_replace, h, w, scale, input_scale, up_scale, up_ft_index, w_edit, w_content, precision):
    if isinstance(path_mask, str):
        mask_base = cv2.imread(path_mask)
    else:
        mask_base = path_mask
    mask_base = cv2.resize(mask_base, (h, w))
    if isinstance(path_mask_replace, str):
        mask_replace = cv2.imread(path_mask_replace)
    else:
        mask_replace = path_mask_replace
    mask_replace = cv2.resize(mask_replace, (h, w))

    dict_mask = {}
    mask_base = img2tensor(mask_base)[0]
    dict_mask['base'] = mask_base
    mask_base = (mask_base>0.5).to('cuda', dtype=precision)
    mask_replace = img2tensor(mask_replace)[0]
    dict_mask['replace'] = mask_replace
    mask_replace = (mask_replace>0.5).to('cuda', dtype=precision)

    mask_base_cur = F.interpolate(mask_base[None,None], (int(mask_base.shape[-2]//scale), int(mask_base.shape[-1]//scale)))>0.5
    mask_replace_cur = F.interpolate(mask_replace[None,None], (int(mask_replace.shape[-2]//scale), int(mask_replace.shape[-1]//scale)))>0.5

    return {
        "dict_mask":dict_mask,
        "mask_base_cur":mask_base_cur,
        "mask_replace_cur":mask_replace_cur,
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "w_edit":w_edit,
        "w_content":w_content,
    }

def process_paste(path_mask, h, w, dx, dy, scale, input_scale, up_scale, up_ft_index, w_edit, w_content, precision, resize_scale=None):
    dx, dy = dx*input_scale, dy*input_scale
    if isinstance(path_mask, str):
        mask_base = cv2.imread(path_mask)
    else:
        mask_base = path_mask
    mask_base = cv2.resize(mask_base, (h, w))

    dict_mask = {}
    mask_base = img2tensor(mask_base)[0][None, None]
    mask_base = (mask_base>0.5).to('cuda', dtype=precision)
    if resize_scale is not None and resize_scale!=1:
        hi, wi = mask_base.shape[-2], mask_base.shape[-1]
        mask_base = F.interpolate(mask_base, (int(hi*resize_scale), int(wi*resize_scale)))
        pad_size_x = np.abs(mask_base.shape[-1]-wi)//2
        pad_size_y = np.abs(mask_base.shape[-2]-hi)//2
        if resize_scale>1:
            mask_base = mask_base[:,:,pad_size_y:pad_size_y+hi,pad_size_x:pad_size_x+wi]
        else:
            temp = torch.zeros(1,1,hi, wi).to(mask_base.device)
            temp[:,:,pad_size_y:pad_size_y+mask_base.shape[-2],pad_size_x:pad_size_x+mask_base.shape[-1]]=mask_base
            mask_base = temp
    mask_replace = mask_base.clone()
    mask_base = torch.roll(mask_base, (int(dy), int(dx)), (-2,-1))
    dict_mask['base'] = mask_base[0,0]
    dict_mask['replace'] = mask_replace[0,0]
    mask_replace = (mask_replace>0.5).to('cuda', dtype=precision)

    mask_base_cur = F.interpolate(mask_base, (int(mask_base.shape[-2]//scale), int(mask_base.shape[-1]//scale)))>0.5
    mask_replace_cur = torch.roll(mask_base_cur, (-int(dy/scale), -int(dx/scale)), (-2,-1))

    return {
        "dict_mask":dict_mask,
        "mask_base_cur":mask_base_cur,
        "mask_replace_cur":mask_replace_cur,
        "up_scale":up_scale,
        "up_ft_index":up_ft_index,
        "w_edit":w_edit,
        "w_content":w_content,
        "w_edit":w_edit,
        "w_content":w_content,
    }