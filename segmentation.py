"""Segmentation routines

This module contains functions for segmentation that are used in the main analysis script.
"""

import numpy as np
import skimage
from scipy import ndimage
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def segment_tail(image):
    """Segment tail region

    segment_tail(image) will segment the tail region and return a binary mask. image should be a single z-plane image containing both the mitochondria and mtDNA labels.
    """
    
    # Combine both images
    combined_image = skimage.exposure.equalize_adapthist(image[:, :, 0].squeeze()) + skimage.exposure.equalize_adapthist(image[:, :, 1].squeeze())

    # Clean the image by subtracting the background
    combined_image_cleaned = skimage.morphology.white_tophat(combined_image, footprint=skimage.morphology.disk(60))
    combined_image_cleaned = skimage.filters.median(combined_image_cleaned, footprint=np.ones((3, 3)), mode='nearest')

    # Generate the mask using a convex hull
    thLvl = skimage.filters.threshold_li(combined_image_cleaned)
    mask_tail = combined_image_cleaned > (0.9 * thLvl)
    
    mask_tail = skimage.morphology.remove_small_holes(mask_tail, 500)
    mask_tail = ndimage.binary_fill_holes(mask_tail)

    mask_tail = skimage.morphology.remove_small_objects(mask_tail, 2500)

    #plt.imshow(mask_tail)
    
    mask_tail_chull = skimage.morphology.convex_hull_image(mask_tail)

    return mask_tail_chull

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

    

    

    

    
