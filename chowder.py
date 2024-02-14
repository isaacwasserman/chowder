import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import imageio
from IPython.display import Image
import tqdm
from PIL import Image
import torch
from ultralytics import FastSAM
from morphing import *
import io
import base64

fastsam = FastSAM("./weights/FastSAM.pt")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

def segment(input, maximum_masks=10):
    results = fastsam(input, device=device, retina_masks=True)
    masks = results[0].masks.data
    masks = masks.squeeze(0).type(torch.bool).cpu().numpy()
    # remove channels with no mask
    # sort by mask area, descending
    mask_areas = [mask.sum() for mask in masks]
    masks = masks[np.any(masks, axis=(1, 2)), :, :]
    masks = masks[np.argsort(mask_areas)[::-1]]
    masks = masks[:maximum_masks]
    for i in range(len(masks)):
        for j in range(len(masks)):
            if i != j:
                masks[i] = masks[i] & ~masks[j]
    empty_mask_indices = []
    for i in range(len(masks)):
        if masks[i].sum() == 0:
            empty_mask_indices.append(i)
    masks = np.delete(masks, empty_mask_indices, axis=0)
    return masks

def texture_from_region(image, mask, index=0):
    pixels = np.where(mask)
    miny = pixels[0].min()
    minx = pixels[1].min()
    maxy = pixels[0].max()
    maxx = pixels[1].max()
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    largest_contour_area = 0
    largest_contour_index = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > largest_contour_area:
            largest_contour_area = area
            largest_contour = contour.reshape(-1, 2)
            largest_contour_index = i
    
    # Reduce the number of points in the contour
    try:
        largest_contour = cv2.approxPolyDP(largest_contour, 1, True)
        largest_contour = largest_contour.reshape(-1, 2)
    except:
        return False

    corners = [(0,0), (0, mask.shape[0]), (mask.shape[1], mask.shape[0]), (mask.shape[1], 0)]
    closest_pts = {corner: None for corner in corners}
    closest_pts_indices = {corner: None for corner in corners}
    for corner in corners:
        diffs = largest_contour - np.array(corner)
        distances = np.linalg.norm(diffs, axis=1)
        closest_pt = largest_contour[np.argmin(distances)]
        closest_pt_index = np.argmin(distances)
        closest_pts[corner] = closest_pt
        closest_pts_indices[corner] = closest_pt_index

    index_of_top_left_point = closest_pts_indices[corners[0]]
    index_of_bottom_left_point = closest_pts_indices[corners[1]]
    index_of_bottom_right_point = closest_pts_indices[corners[2]]
    index_of_top_right_point = closest_pts_indices[corners[3]]

    rolled_contour_points = np.roll(largest_contour, -index_of_top_left_point, axis=0)
    index_of_top_left_point = np.where((rolled_contour_points == closest_pts[corners[0]]).all(axis=1))[0][0]
    index_of_bottom_left_point = np.where((rolled_contour_points == closest_pts[corners[1]]).all(axis=1))[0][0]
    index_of_bottom_right_point = np.where((rolled_contour_points == closest_pts[corners[2]]).all(axis=1))[0][0]
    index_of_top_right_point = np.where((rolled_contour_points == closest_pts[corners[3]]).all(axis=1))[0][0]

    left_points = rolled_contour_points[index_of_top_left_point:index_of_bottom_left_point + 1]
    bottom_points = rolled_contour_points[index_of_bottom_left_point:index_of_bottom_right_point + 1]
    right_points = rolled_contour_points[index_of_bottom_right_point:index_of_top_right_point + 1]
    top_points = np.concatenate([rolled_contour_points[index_of_top_right_point:], rolled_contour_points[:index_of_top_left_point + 1], rolled_contour_points[0:1]])

    src_points = np.concatenate([left_points[:-1], bottom_points[:-1], right_points[:-1], top_points[:-1]])
    try:
        dst_points = np.concatenate([
            np.linspace((0, 0), (0, mask.shape[0]), num=left_points.shape[0] - 1),
            np.linspace((0, mask.shape[0]), (mask.shape[1], mask.shape[0]), num=bottom_points.shape[0] - 1),
            np.linspace((mask.shape[1], mask.shape[0]), (mask.shape[1], 0), num=right_points.shape[0] - 1),
            np.linspace((mask.shape[1], 0), (0, 0), num=top_points.shape[0] - 1)
        ])
    except:
        dst_points = src_points

    morphed = ImageMorphingTriangulation(image, image, src_points, dst_points, 1, 0)
    morphed = np.array(Image.fromarray(morphed).resize((maxx - minx, maxy - miny)))
    return morphed

def selective_scramble_frame(image, textures, masks, regions_per_frame=1, index=0, scramble=True):
    canvas = image.copy()
    num_samples = min([regions_per_frame, len(textures), len(masks)])
    src_textures_indices = np.random.choice(len(textures), num_samples, replace=False)
    tgt_masks_indices = np.random.choice(len(masks), num_samples, replace=False)
    for i in range(num_samples):
        tgt_pixels = np.where(masks[tgt_masks_indices[i]])
        tgt_miny = tgt_pixels[0].min()
        tgt_minx = tgt_pixels[1].min()
        tgt_maxy = tgt_pixels[0].max()
        tgt_maxx = tgt_pixels[1].max()
        cropped_tgt_pixels = (tgt_pixels[0] - tgt_miny, tgt_pixels[1] - tgt_minx)
        texture = np.array(Image.fromarray(textures[src_textures_indices[i]]).resize((tgt_maxx - tgt_minx + 1, tgt_maxy - tgt_miny + 1)))
        texture_pixels = texture[cropped_tgt_pixels]
        canvas[tgt_pixels] = texture_pixels
    return canvas

def selective_scramble(image, masks, num_frames=1, regions_per_frame=1):
    frames = []
    textures = []
    for mask in tqdm.tqdm(masks):
        texture = texture_from_region(image, mask)
        if texture is not False:
            textures.append(texture)
    for frame_index in range(num_frames):
        canvas = selective_scramble_frame(image, textures, masks, regions_per_frame=regions_per_frame)
        frames.append(canvas)
    return frames

def selective_reveal(image, masks, num_frames=1, regions_per_frame=1):
    frames = []
    for frame_index in tqdm.tqdm(range(num_frames)):
        canvas = np.zeros_like(image)
        regions_chosen = np.random.choice(len(masks), regions_per_frame, replace=False)
        for i in regions_chosen:
            element_mask = masks[i]
            canvas[element_mask] = image[element_mask]
        frames.append(canvas)
    return frames

def frames_to_gif(frames, fps=10, path=None):
    # Create GIF in memory
    with io.BytesIO() as gif_bytes:
        duration = 1000 / fps
        imageio.mimsave(gif_bytes, frames, format='GIF', duration=duration, loop=0)
        
        # Read bytes    
        gif_blob = gif_bytes.getvalue()

        if path is not None:
            with open(path, 'wb') as f:
                f.write(gif_blob)
        else:
            # Encode blob as base64 dataURL
            gif_base64 = base64.b64encode(gif_blob).decode('utf-8')
            
            return gif_base64