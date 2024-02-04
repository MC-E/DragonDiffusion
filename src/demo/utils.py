import numpy as np
import gradio as gr
import cv2
from copy import deepcopy
import torch
from torchvision import transforms
from PIL import Image

from sam.efficient_sam.build_efficient_sam import build_efficient_sam_vits
from src.utils.utils import resize_numpy_image

sam = build_efficient_sam_vits()

def show_point_or_box(image, global_points):
    # for point
    if len(global_points) == 1:
        image = cv2.circle(image, global_points[0], 10, (0, 0, 255), -1)
    # for box
    if len(global_points) == 2:
        p1 = global_points[0]
        p2 = global_points[1]
        image = cv2.rectangle(image,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(0,0,255),2)

    return image
    
def segment_with_points(
    image,
    original_image,
    global_points,
    global_point_label,
    evt: gr.SelectData,
    img_direction,
):
    if original_image is None:
        original_image = image
    else:
        image = original_image
    if img_direction is None:
        img_direction = original_image
    x, y = evt.index[0], evt.index[1]
    if len(global_points) == 0:
        global_points.append([x, y])
        global_point_label.append(2)
        image_with_point= show_point_or_box(image.copy(), global_points)
        return image_with_point, original_image, None, global_points, global_point_label, img_direction, original_image
    elif len(global_points) == 1:
        global_points.append([x, y])
        global_point_label.append(3)
        x1, y1 = global_points[0]
        x2, y2 = global_points[1]
        if x1 < x2 and y1 >= y2:
            global_points[0][0] = x1
            global_points[0][1] = y2
            global_points[1][0] = x2
            global_points[1][1] = y1
        elif x1 >= x2 and y1 < y2:
            global_points[0][0] = x2
            global_points[0][1] = y1
            global_points[1][0] = x1
            global_points[1][1] = y2
        elif x1 >= x2 and y1 >= y2:
            global_points[0][0] = x2
            global_points[0][1] = y2
            global_points[1][0] = x1
            global_points[1][1] = y1
        image_with_point = show_point_or_box(image.copy(), global_points)
        # data process
        input_point = np.array(global_points)
        input_label = np.array(global_point_label)
        pts_sampled = torch.reshape(torch.tensor(input_point), [1, 1, -1, 2])
        pts_labels = torch.reshape(torch.tensor(input_label), [1, 1, -1])
        img_tensor = transforms.ToTensor()(image)
        # sam
        predicted_logits, predicted_iou = sam(
            img_tensor[None, ...],
            pts_sampled,
            pts_labels,
        )
        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).float().cpu().detach().numpy()
        mask_image = (mask*255.).astype(np.uint8)
        return image_with_point, original_image, mask_image, global_points, global_point_label, img_direction, original_image
    else:
        global_points=[[x, y]]
        global_point_label=[2]
        image_with_point= show_point_or_box(image.copy(), global_points)
        return image_with_point, original_image, None, global_points, global_point_label, img_direction, original_image


def segment_with_points_paste(
    image,
    original_image,
    global_points,
    global_point_label,
    image_b,
    evt: gr.SelectData,
    dx, 
    dy, 
    resize_scale

):
    if original_image is None:
        original_image = image
    else:
        image = original_image
    x, y = evt.index[0], evt.index[1]
    if len(global_points) == 0:
        global_points.append([x, y])
        global_point_label.append(2)
        image_with_point= show_point_or_box(image.copy(), global_points)
        return image_with_point, original_image, None, global_points, global_point_label, None
    elif len(global_points) == 1:
        global_points.append([x, y])
        global_point_label.append(3)
        x1, y1 = global_points[0]
        x2, y2 = global_points[1]
        if x1 < x2 and y1 >= y2:
            global_points[0][0] = x1
            global_points[0][1] = y2
            global_points[1][0] = x2
            global_points[1][1] = y1
        elif x1 >= x2 and y1 < y2:
            global_points[0][0] = x2
            global_points[0][1] = y1
            global_points[1][0] = x1
            global_points[1][1] = y2
        elif x1 >= x2 and y1 >= y2:
            global_points[0][0] = x2
            global_points[0][1] = y2
            global_points[1][0] = x1
            global_points[1][1] = y1
        image_with_point = show_point_or_box(image.copy(), global_points)
        # data process
        input_point = np.array(global_points)
        input_label = np.array(global_point_label)
        pts_sampled = torch.reshape(torch.tensor(input_point), [1, 1, -1, 2])
        pts_labels = torch.reshape(torch.tensor(input_label), [1, 1, -1])
        img_tensor = transforms.ToTensor()(image)
        # sam
        predicted_logits, predicted_iou = sam(
            img_tensor[None, ...],
            pts_sampled,
            pts_labels,
        )
        mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).float().cpu().detach().numpy()
        mask_uint8 = (mask*255.).astype(np.uint8)

        return image_with_point, original_image, paste_with_mask_and_offset(image, image_b, mask_uint8, dx, dy, resize_scale), global_points, global_point_label, mask_uint8
    else:
        global_points=[[x, y]]
        global_point_label=[2]
        image_with_point= show_point_or_box(image.copy(), global_points)
        return image_with_point, original_image, None, global_points, global_point_label, None

def paste_with_mask_and_offset(image_a, image_b, mask, x_offset=0, y_offset=0, delta=1):
    try:
        numpy_mask = np.array(mask)
        y_coords, x_coords = np.nonzero(numpy_mask)  
        x_min = x_coords.min()  
        x_max = x_coords.max()  
        y_min = y_coords.min()  
        y_max = y_coords.max()
        target_center_x = int((x_min + x_max) / 2)
        target_center_y = int((y_min + y_max) / 2)

        image_a = Image.fromarray(image_a)
        image_b = Image.fromarray(image_b)
        mask = Image.fromarray(mask)

        if image_a.size != mask.size:
            mask = mask.resize(image_a.size)

        cropped_image = Image.composite(image_a, Image.new('RGBA', image_a.size, (0, 0, 0, 0)), mask)
        x_b = int(target_center_x * (image_b.width / cropped_image.width))
        y_b = int(target_center_y * (image_b.height / cropped_image.height))
        x_offset = x_offset - int((delta - 1) * x_b)
        y_offset = y_offset - int((delta - 1) * y_b)
        cropped_image = cropped_image.resize(image_b.size)
        new_size = (int(cropped_image.width * delta), int(cropped_image.height * delta))
        cropped_image = cropped_image.resize(new_size)
        image_b.putalpha(128) 
        result_image = Image.new('RGBA', image_b.size, (0, 0, 0, 0))
        result_image.paste(image_b, (0, 0))
        result_image.paste(cropped_image, (x_offset, y_offset), mask=cropped_image)

        return result_image
    except:
        return None

def upload_image_move(img, original_image):
    if original_image is not None:
        return original_image
    else:
        return img

def fun_clear(*args):
    result = []
    for arg in args:
        if isinstance(arg, list):
            result.append([])
        else:
            result.append(None)
    return tuple(result)

def clear_points(img):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()

    return [], masked_img

def get_point(img, sel_pix, evt: gr.SelectData):
    sel_pix.append(evt.index)
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        else:
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        points.append(tuple(point))
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

def get_point_move(original_image, img, sel_pix, evt: gr.SelectData):
    if original_image is not None:
        img = original_image.copy()
    else:
        original_image = img.copy()
    if len(sel_pix)<2:
        sel_pix.append(evt.index)
    else:
        sel_pix = [evt.index]
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        else:
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        points.append(tuple(point))
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    img = np.array(img)

    return img, original_image, sel_pix

def store_img(img):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()

    return image, masked_img, mask

def store_img_move(img):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()

    return image, masked_img, (mask*255.).astype(np.uint8)

def mask_image(image, mask, color=[255,0,0], alpha=0.5, max_resolution=None):
    """ Overlay mask on image for visualization purpose. 
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    if max_resolution is not None:
        image, _ = resize_numpy_image(image, max_resolution*max_resolution)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),interpolation=cv2.INTER_NEAREST)

    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    contours = cv2.findContours(np.uint8(deepcopy(mask)), cv2.RETR_TREE, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return out