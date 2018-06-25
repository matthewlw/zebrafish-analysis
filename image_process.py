import skimage
import skimage.util
import skimage.measure
import skimage.morphology
import numpy as np

def brightest_portion(image, fraction):
    inv = skimage.util.invert(image)
    cutoff = np.sort(inv.ravel())[int((1-fraction) * image.size)]
    return inv >= cutoff

def remove_tiny_blobs(image, tiny_threshold=20):
    return skimage.morphology.remove_small_objects(image, min_size=tiny_threshold)

def crop_to_nonzero(image, border=2):
    (ys, xs) = image.nonzero()
    return image[min(ys)-border:max(ys)+border, min(xs)-border:max(xs)+border]

def take_biggest_blob(image):
    new_image = np.zeros_like(image)
    integered = skimage.img_as_int(image)
    labeled = skimage.measure.label(integered)
    blobs = skimage.measure.regionprops(labeled, intensity_image=integered, cache=True)
    blobs_by_area = sorted(blobs, key=lambda blob: blob.area, reverse=True)
    biggest_blob = blobs_by_area[0]
    for [r, c] in biggest_blob.coords:
        new_image[r, c] = True
    return new_image
