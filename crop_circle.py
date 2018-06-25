import sys
import skimage
import skimage.filters
import skimage.transform
import skimage.feature
import skimage.draw
import numpy as np

def get_circle_params(image, radius_adjust=0, low=300, high=500):
    thresh = skimage.filters.threshold_otsu(image)
    binary = image >= thresh
    edges = skimage.feature.canny(
        skimage.img_as_ubyte(image),
        sigma=3,
        low_threshold=10,
        high_threshold=50
    )
    hough_radii = np.arange(300, 500, 2)
    hough_res = skimage.transform.hough_circle(edges, hough_radii)
    [accums], [cx], [cy], [radius] = skimage.transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
    return (cx, cy, radius)

def get_circle_coords(image, radius_adjust=0, low=300, high=500):
    (cx, cy, radius) = get_circle_params(image, radius_adjust, low, high)
    crr, ccc = skimage.draw.circle(cy, cx, radius + radius_adjust, image.shape)
    return (crr, ccc)

def crop_circle(image, radius_adjust=0, low=300, high=500):
    (cx, cy, radius) = get_circle_params(image, radius_adjust, low, high)
    crr, ccc = skimage.draw.circle(cy, cx, radius + radius_adjust, image.shape)
    new_image = np.zeros_like(image)
    new_image[crr, ccc] = image[crr, ccc]
    return new_image
