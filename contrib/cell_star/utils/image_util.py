# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import os
from os import makedirs
from os.path import  exists

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.misc
import scipy.ndimage
from numpy import argwhere
from scipy.ndimage.filters import *

debug_image_path = "debug"

SHOW = False
SILENCE = False

def prepare_debug_folder():
    if not exists(debug_image_path):
        makedirs(debug_image_path)

def convolve2d(img, kernel, mode='same'):
    return convolve(img, kernel)


def extend_slices(my_slices, extension):
    def extend_slice(my_slice, extend):
        ind = (max(0, my_slice.indices(100000)[0] - extend), my_slice.indices(100000)[1] + extend)
        return slice(*ind)

    return extend_slice(my_slices[0], extension), extend_slice(my_slices[1], extension)


def get_bounding_box(image_mask):
    """
    Calculates the minimal bounding box for non zero elements.
    @returns [ystart,ystop), [xstart,xstop) or None, None
    """
    non_zero_points = argwhere(image_mask)
    if len(non_zero_points) == 0:
        return None
    (ystart, xstart), (ystop, xstop) = non_zero_points.min(0), non_zero_points.max(0) + 1
    return (ystart, ystop), (xstart, xstop)


def get_circle_kernel(radius):
    """
    Creates radius x radius bool image of the circle.
    @param radius: radius of the circle
    """
    y, x = np.ogrid[np.floor(-radius):np.ceil(radius) + 1, np.floor(-radius):np.ceil(radius) + 1]
    return x ** 2 + y ** 2 <= radius ** 2


def image_dilate(image, radius):
    image = np.copy(image)
    if radius <= 1:
        return image

    box = get_bounding_box(image)
    if box is None:
        return image
    ys, xs = box
    lp, hp = contain_pixel(image.shape, (ys[0] - radius, xs[0] - radius)), \
        contain_pixel(image.shape, (ys[1] + radius, xs[1] + radius))
    ys, xs = (lp[0], hp[0]), (lp[1], hp[1])
    morphology_element = get_circle_kernel(radius)
    dilated_part = sp.ndimage.morphology.binary_dilation(image[ys[0]:ys[1], xs[0]:xs[1]], morphology_element)
    image[ys[0]:ys[1], xs[0]:xs[1]] = dilated_part
    return image


def image_dilate_with_element(image, n):
    return sp.ndimage.morphology.grey_dilation(image, size=(n, n))


def image_erode(image, radius):
    morphology_element = get_circle_kernel(radius)
    return sp.ndimage.morphology.binary_erosion(image, morphology_element)


def fill_foreground_holes(mask, kernel_size, minimal_hole_size, min_cluster_area_scaled, mask_min_radius_scaled):
    filled_black_holes = fill_holes(mask, kernel_size, minimal_hole_size)

    holes_remaining = np.logical_not(filled_black_holes)
    filled_small_holes = mark_small_areas(holes_remaining, min_cluster_area_scaled, filled_black_holes)

    morphology_enhanced = image_erode(filled_small_holes, mask_min_radius_scaled)
    morphology_enhanced = image_dilate(morphology_enhanced, mask_min_radius_scaled)

    dilated_mask = dilate_big_areas(morphology_enhanced, min_cluster_area_scaled, kernel_size)

    return dilated_mask


def mark_small_areas(mask, max_hole_size, result_mask):
    components, num_components = sp.ndimage.label(mask, np.ones((3, 3)))
    slices = sp.ndimage.find_objects(components)
    for label, slice in zip(range(1, num_components + 1),slices):
        components_slice = components[slice] == label
        if np.count_nonzero(components_slice) < max_hole_size:
            result_mask[slice][components_slice] = True
    return result_mask


def dilate_big_areas(mask, min_area_size, dilate_radius):
    components, num_components = sp.ndimage.label(mask, np.ones((3, 3)))
    component = np.zeros(mask.shape, dtype=bool)
    for label in range(1, num_components + 1):
        np.equal(components, label, component)
        if np.count_nonzero(component) > min_area_size:
            tmp_mask = image_dilate(component, dilate_radius)
            mask = mask | tmp_mask

    return mask


def fill_holes(mask, kernel_size, minimal_hole_size):
    """
    Fills holes in a given mask using iterative close + dilate morphological operations and filtering small patches.
    @param mask: mask which holes are to be filled
    @param kernel_size: size of the morphological element used to dilate/erode mask
    @param minimal_hole_size: holes with area smaller than param are to be removed
    """
    nr = 1
    morphology_element = get_circle_kernel(kernel_size)
    while True:
        new_mask = mask.copy()
        # find connected components
        components, num_components = sp.ndimage.label(np.logical_not(new_mask), np.ones((3, 3)))
        slices = sp.ndimage.find_objects(components)
        for label, slice in zip(range(1, num_components + 1), slices):
            slice = extend_slices(slice,kernel_size * 2)
            components_slice = components[slice] == label
            # filter small components
            if np.count_nonzero(components_slice) < minimal_hole_size:
                new_mask[slice] |= components_slice
            else:
                # shrink components and check if they fell apart
                # close holes
                components_slice = sp.ndimage.morphology.binary_closing(components_slice, morphology_element)

                # erode holes
                components_slice = sp.ndimage.morphology.binary_erosion(components_slice, morphology_element)

                # don't invade masked pixels
                components_slice &= np.logical_not(new_mask[slice])

                # recount connected components and check sizes
                mark_small_areas(components_slice, minimal_hole_size, new_mask[slice])

        # check if it is the fix point
        if (mask == new_mask).all():
            break
        else:
            mask = new_mask

        nr += 1

    return mask


def draw_seeds(seeds, background, title="some_source"):
    if not SILENCE:
        prepare_debug_folder()
        fig = plt.figure("draw_seeds")
        fig.frameon = False
        plt.imshow(background, cmap=plt.cm.gray)
        plt.plot([s.x for s in seeds], [s.y for s in seeds], 'bo', markersize=3)
        plt.savefig(os.path.join(debug_image_path, "seeds_"+title+".png"), pad_inches=0.0)
        fig.clf()
        plt.close(fig)


def contain_pixel(shape, pixel):
    """
    Trims pixel to given dimentions, converts pixel position to int
    @param shape: size (height, width) exclusive
    @param pixel: pixel to push inside shape
    """
    (py, px) = pixel
    (py, px) = ((np.minimum(np.maximum(py + 0.5, 0), shape[0] - 1)).astype(int),
                (np.minimum(np.maximum(px + 0.5, 0), shape[1] - 1)).astype(int))
    return py, px


def find_maxima(image):
    """
    Finds local maxima in given image
    @param image: image from which maxima will be found
    """
    height = image.shape[0]
    width = image.shape[1]

    right = np.zeros(image.shape)
    left = np.zeros(image.shape)
    up = np.zeros(image.shape)
    down = np.zeros(image.shape)

    epsilon = 0.00  #0001

    right[0:height, 0:width - 1] = np.array(image[:, 0:width - 1] - image[:, 1:width] > epsilon)
    left[0:height, 1:width] = np.array(image[:, 1:width] - image[:, 0:width - 1] > epsilon)
    up[0:height - 1, 0:width] = np.array(image[0:height - 1, :] - image[1:height, :] > epsilon)
    down[1:height, 0:width] = np.array(image[1:height, :] - image[0:height - 1, :] > epsilon)

    return right * left * up * down


def exclude_segments(image, segments, val):
    """
    Sets exclusion value for given segments in given image
    @param image: image from which segments will be excluded
    @param segments: segments to be excluded from image
    @param val: value to be set in segments as exclusion value
    """
    segment_mask = segments > 0
    inverted_segment_mask = np.logical_not(segment_mask)
    image_segments_zeroed = image * inverted_segment_mask
    image_segments_valued = image_segments_zeroed + (segment_mask * val)

    return image_segments_valued

def image_median_filter(image, size):
    if size < 1:
        return image

    return median_filter(image, (size, size))

def image_blur(image, times):
    """
    Performs image blur with kernel: [[2, 3, 2], [3, 12, 3], [2, 3, 2]] / 32
    @param image: image to be blurred (assumed as numpy.array of values from 0 to 1)
    @param times: specifies how many times blurring will be performed
    """
    kernel = np.array([[2, 3, 2], [3, 12, 3], [2, 3, 2]]) / 32.0
    blurred = convolve2d(image, kernel, 'same')

    for _ in xrange(int(times) - 1):
        blurred = convolve2d(blurred, kernel, 'same')

    return blurred


def image_smooth(image, radius):
    """
    Performs image blur with circular kernel.
    @param image: image to be blurred (assumed as numpy.array of values from 0 to 1)
    @param radius: radius of the kernel
    """
    if radius < 1:
        return image

    kernel = get_circle_kernel(radius).astype(float)
    kernel /= np.sum(kernel)
    image = np.array(image, dtype=float)
    return convolve2d(image, kernel, 'same')


def image_normalize(image):
    """
    Performs image normalization (vide: matlab mat2gray)
    @param image: image to be normalized (assumed as numpy.array of values from 0 to 1)
    """
    minimum = np.amin(image)
    maximum = np.amax(image)

    delta = 1
    if maximum != minimum:
        delta = 1 / (maximum - minimum)
    shift = - minimum * delta

    image_normalized = delta * image + shift

    return np.minimum(np.maximum(image_normalized, 0), 1)


def image_save(image, title):
    """
    Displays image with title using matplotlib.pyplot
    @param image:
    @param title:
    """

    if not SILENCE:
        prepare_debug_folder()
        sp.misc.imsave(os.path.join(debug_image_path, title + '.png'), image)


def image_show(image, title):
    """
    Displays image with title using matplotlib.pyplot
    @param image:
    @param title:
    """
    if not SILENCE and SHOW:
        prepare_debug_folder()
        fig = plt.figure(title)
        plt.imshow(image, cmap=plt.cm.gray, interpolation='none')
        plt.show()
        fig.clf()
        plt.close(fig)


def draw_overlay(image, x, y):
    if not SILENCE and SHOW:
        prepare_debug_folder()
        fig = plt.figure()
        plt.imshow(image, cmap=plt.cm.gray, interpolation='none')
        plt.plot(x, y)
        plt.show()
        fig.clf()
        plt.close(fig)


def draw_snakes(image, snakes, outliers=.1, it=0):
    if not SILENCE and len(snakes) > 1:
        prepare_debug_folder()
        snakes = sorted(snakes, key=lambda ss: ss.rank)
        fig = plt.figure("draw_snakes")
        plt.imshow(image, cmap=plt.cm.gray, interpolation='none')

        snakes_tc = snakes[:int(len(snakes) * (1 - outliers))]

        max_rank = snakes_tc[-1].rank
        min_rank = snakes_tc[0].rank
        rank_range = max_rank - min_rank
        if rank_range == 0:  # for example there is one snake
            rank_range = max_rank

        rank_ci = lambda rank: 999 * ((rank - min_rank) / rank_range) if rank <= max_rank else 999
        colors = plt.cm.jet(np.linspace(0, 1, 1000))
        s_colors = [colors[rank_ci(s.rank)] for s in snakes]

        for snake, color in zip(snakes, s_colors):
            plt.plot(snake.xs, snake.ys, c=color, linewidth=4.0)

        plt.savefig(os.path.join(debug_image_path, "snakes_rainbow_"+str(it)+".png"), pad_inches=0.0)
        if SHOW:
            plt.show()

        fig.clf()
        plt.close(fig)


def tiff16_to_float(image):
    image = np.array(image, dtype=float)
    return (image - image.min()) / image.max()


def set_image_border(image, val):
    """
    Sets pixel values at image borders to given value
    @param image: image that borders will be set to given value
    @param val: value to be set
    """
    image[0, :] = val
    image[:, 0] = val
    image[image.shape[0] - 1, :] = val
    image[:, image.shape[1] - 1] = val

    return image


def paths(test_path):
    return os.path.join(test_path, 'test_reference'), os.path.join(test_path, 'frames'), os.path.join(test_path, 'background')


def frame_exists(frames_path, frame):
    return os.path.exists(frames_path + frame)


def load_ref_image(test_reference_path, frame, name):
    return load_image(test_reference_path + frame + "/" + name + ".tif")


def load_frame(frames_path, filename):
    return load_image(frames_path + filename)


def load_frame_background(background_path, filename):
    try:
        image = load_image(background_path + filename)
        if image is None:
            return None

        return image
    except IOError:
        return None


def load_image(filename, scaling=True):
    if filename == '':
        return None
    image = scipy.misc.imread(filename)
    if image.max() > 1 and scaling:
        image = np.array(image, dtype=float) / np.iinfo(image.dtype).max
    if image.size == 1:
        image = image.item()
        width = image.size[0]
        height = image.size[1]
        image1d = np.array(list(image.getdata()))
        image2d = np.zeros(image.size)

        for y in xrange(height):
            for x in xrange(width):
                image2d[y,x] = image1d[y*width + x]
    else:
        image2d = image.astype(float)

    return image2d
