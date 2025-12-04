"""Segmentation routines

This module contains functions for segmentation that are used in the main analysis script.
"""

import numpy as np
import skimage
from scipy import ndimage
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scipy import signal

def segment_tail(image):
    """Segment tail region

    segment_tail(image) will segment the tail region and return a binary mask. image should be a single z-plane image containing both the mitochondria and mtDNA labels.
    """

    sam2_checkpoint = "inc/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    
    predictor = SAM2ImagePredictor(sam2_model)

    image_norm = image[:, :, 1]
    image_norm = image_norm.astype(float)
    image_norm = (image_norm - np.min(image_norm))/(np.max(image_norm) - np.min(image_norm))
    
    image_filtered = skimage.filters.gaussian(image_norm, 2)
    
    intensity_hist, bin_edges = np.histogram(image_filtered.ravel(), bins=256)
    
    pks, _ = signal.find_peaks(intensity_hist, height=1000, distance = 5)

    thresh = 0.95 * bin_edges[pks[-1]]

    mask_tail = image_filtered > thresh
    

    # # Auto-segment the tail by first using thresholding to identify the tail region
    # combined_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    # for z in range(image.shape[2]):
    #     curr_slice = image[:, :, z].astype(np.float32)
    #     curr_slice = (curr_slice - np.min(curr_slice))/(np.max(curr_slice) - np.min(curr_slice))
    #     combined_image[:, :, z] = curr_slice
    
    # combined_image_gray = skimage.color.rgb2gray(combined_image)
    
    # # Clean the image by subtracting the background
    # combined_image_cleaned = skimage.filters.gaussian(combined_image_gray, 1)

    # # Threshold the tail
    # thLvl = skimage.filters.threshold_li(combined_image_cleaned)
    # mask_tail = combined_image_cleaned > thLvl
    # mask_tail = ndimage.binary_fill_holes(mask_tail)
    # mask_tail = skimage.morphology.remove_small_objects(mask_tail, 500)
    # mask_tail = skimage.morphology.binary_opening(mask_tail, skimage.morphology.disk(4))

    # plt.imshow(mask_tail)

    # Measure the bounding box
    tail_labels = mask_tail.astype(int)    
    props = skimage.measure.regionprops(tail_labels)

    # Run SAM2 but only on the green channel
    image_input = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    curr_slice = image[:, :, 1].astype(np.float32)
    curr_slice = skimage.filters.gaussian(curr_slice, 1)
    
    curr_slice = (curr_slice - np.min(curr_slice))/(np.max(curr_slice) - np.min(curr_slice))
    #curr_slice = skimage.exposure.adjust_sigmoid(curr_slice)
    
    image_input[:, :, 1] = curr_slice
    
    predictor.set_image(image_input)
    
    bbox = np.array([props[0].bbox[1], props[0].bbox[0], props[0].bbox[3], props[0].bbox[2]])  #[left, top, right, bottom]
    
    # Increase the bbox a little, but clamp it to the size of the image
    px_increase = 20
    
    bbox[0] = np.clip(bbox[0] - px_increase, 0, None)
    bbox[1] = np.clip(bbox[1] - px_increase, 0, None)
    bbox[2] = np.clip(bbox[2] + px_increase, None, image.shape[1])
    bbox[3] = np.clip(bbox[3] + px_increase, None, image.shape[0])
    
    # print(bbox)
    
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox,
        multimask_output=True,
    )
    
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    # scores = scores[sorted_ind]

    best_mask = masks[0, :, :] > 0
    best_mask = skimage.morphology.remove_small_objects(best_mask, 500)

    return best_mask

def segment_mitochondria(image):
    """Segment mitochondrial network

    segment_mitochondria(image) will segment the mitochondrial region and return a binary mask. image should be a single z-plane image containing both the mitochondria and mtDNA labels.
    """

    image_mito = image[:, :, 0].squeeze()

    filtImage = skimage.morphology.white_tophat(image_mito, skimage.morphology.disk(50))

    thresh = skimage.filters.threshold_li(filtImage)

    mask = filtImage >= thresh
    mask = skimage.morphology.binary_opening(mask, skimage.morphology.disk(2))

    mask = skimage.morphology.remove_small_objects(mask, 50)

    return mask

def segment_spots(image):

    currSpotImg = image[:, :, 1].squeeze()

    currSpotImg = skimage.morphology.white_tophat(currSpotImg, skimage.morphology.disk(50))
    
    diffGauss = skimage.filters.difference_of_gaussians(currSpotImg, 2, 7)
    
    spotMask = diffGauss >= 0.0001

    return spotMask

    

    

    

    
