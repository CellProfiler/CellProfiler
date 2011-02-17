"""Threshold.py - code for thresholding images

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

# The help text for various thresholding options whose code resides here is in modules/identify.py

__version__="$Revision$"

import inspect
import math
import numpy as np
import scipy.ndimage
import scipy.sparse

from cellprofiler.cpmath.otsu import otsu, entropy, otsu3, entropy3
from cellprofiler.cpmath.smooth import smooth_with_noise
from cellprofiler.cpmath.filter import stretch, unstretch

TM_OTSU                         = "Otsu"
TM_OTSU_GLOBAL                  = "Otsu Global"
TM_OTSU_ADAPTIVE                = "Otsu Adaptive"
TM_OTSU_PER_OBJECT              = "Otsu PerObject"
TM_MOG                          = "MoG"
TM_MOG_GLOBAL                   = "MoG Global"
TM_MOG_ADAPTIVE                 = "MoG Adaptive"
TM_MOG_PER_OBJECT               = "MoG PerObject"
TM_BACKGROUND                   = "Background"
TM_BACKGROUND_GLOBAL            = "Background Global"
TM_BACKGROUND_ADAPTIVE          = "Background Adaptive"
TM_BACKGROUND_PER_OBJECT        = "Background PerObject"
TM_ROBUST_BACKGROUND            = "RobustBackground"
TM_ROBUST_BACKGROUND_GLOBAL     = "RobustBackground Global"
TM_ROBUST_BACKGROUND_ADAPTIVE   = "RobustBackground Adaptive"
TM_ROBUST_BACKGROUND_PER_OBJECT = "RobustBackground PerObject"
TM_RIDLER_CALVARD               = "RidlerCalvard"
TM_RIDLER_CALVARD_GLOBAL        = "RidlerCalvard Global"
TM_RIDLER_CALVARD_ADAPTIVE      = "RidlerCalvard Adaptive"
TM_RIDLER_CALVARD_PER_OBJECT    = "RidlerCalvard PerObject"
TM_KAPUR                        = "Kapur"
TM_KAPUR_GLOBAL                 = "Kapur Global"
TM_KAPUR_ADAPTIVE               = "Kapur Adaptive"
TM_KAPUR_PER_OBJECT             = "Kapur PerObject"
TM_MANUAL                       = "Manual"
TM_MEASUREMENT                  = "Measurement"
TM_BINARY_IMAGE                 = "Binary image"
'''Compute a single threshold for the entire image'''
TM_GLOBAL                       = "Global"

'''Compute a local thresholding matrix of the same size as the image'''
TM_ADAPTIVE                     = "Adaptive"

'''Compute a threshold for each labeled object in the image'''
TM_PER_OBJECT                   = "PerObject"

TM_METHODS =  [TM_OTSU_GLOBAL, TM_OTSU_ADAPTIVE, TM_OTSU_PER_OBJECT,
               TM_MOG_GLOBAL, TM_MOG_ADAPTIVE, TM_MOG_PER_OBJECT,
               TM_BACKGROUND_GLOBAL, TM_BACKGROUND_ADAPTIVE, TM_BACKGROUND_PER_OBJECT,
               TM_ROBUST_BACKGROUND_GLOBAL, TM_ROBUST_BACKGROUND_ADAPTIVE, TM_ROBUST_BACKGROUND_PER_OBJECT,
               TM_RIDLER_CALVARD_GLOBAL, TM_RIDLER_CALVARD_ADAPTIVE, TM_RIDLER_CALVARD_PER_OBJECT,
               TM_KAPUR_GLOBAL, TM_KAPUR_ADAPTIVE, TM_KAPUR_PER_OBJECT,
               TM_MANUAL, TM_BINARY_IMAGE, TM_MEASUREMENT]

TM_GLOBAL_METHODS = [TM_OTSU_GLOBAL, TM_MOG_GLOBAL, TM_BACKGROUND_GLOBAL,
                     TM_ROBUST_BACKGROUND_GLOBAL, TM_RIDLER_CALVARD_GLOBAL, 
                     TM_KAPUR_GLOBAL ]

def get_threshold(threshold_method, threshold_modifier, image, 
                  mask=None, labels = None,
                  threshold_range_min = None, threshold_range_max = None,
                  threshold_correction_factor = 1.0,
                  object_fraction = 0.2,
                  two_class_otsu = True,
                  use_weighted_variance = True,
                  assign_middle_to_foreground = True):
    """Compute a threshold for an image
    
    threshold_method - one of the TM_ methods above
    threshold_modifier - TM_GLOBAL to calculate one threshold over entire image
                         TM_ADAPTIVE to calculate a per-pixel threshold
                         TM_PER_OBJECT to calculate a different threshold for
                         each object
    image - a NxM numpy array of the image data

    Returns a tuple of local_threshold and global_threshold where:
    * global_threshold is the single number calculated using the threshold
    method over the whole image
    * local_threshold is the global_threshold for global methods. For adaptive
    and per-object thresholding, local_threshold is a matrix of threshold
    values representing the threshold to be applied at each pixel of the
    image.
    
    Different methods have optional and required parameters:
    Required:
    TM_PER_OBJECT:
    labels - a labels matrix that defines the extents of the individual objects
             to be thresholded separately.
    
    Optional:
    All:
    mask - a mask of the significant pixels in the image
    threshold_range_min, threshold_range_max - constrain the threshold
        values to be examined to values between these limits
    threshold_correction_factor - the calculated threshold is multiplied
        by this number to get the final threshold
    TM_MOG (mixture of Gaussians):
    object_fraction - fraction of image expected to be occupied by objects
        (pixels that are above the threshold)
    TM_OTSU - We have algorithms derived from Otsu. There is a three-class
              version of Otsu in addition to the two class. There is also
              an entropy measure in addition to the weighted variance.
              two_class_otsu - assume that the distribution represents
                               two intensity classes if true, three if false.
              use_weighted_variance - use Otsu's weighted variance if true,
                                      an entropy measure if false
              assign_middle_to_foreground - assign pixels in the middle class
                               in a three-class Otsu to the foreground if true
                               or the background if false.
    """
    global_threshold = get_global_threshold(threshold_method, image, mask, 
                                            object_fraction,
                                            two_class_otsu,
                                            use_weighted_variance,
                                            assign_middle_to_foreground)
    global_threshold *= threshold_correction_factor
    if not threshold_range_min is None:
        global_threshold = max(global_threshold, threshold_range_min)
    if not threshold_range_max is None:
        global_threshold = min(global_threshold, threshold_range_max)
    if threshold_modifier == TM_GLOBAL:
        local_threshold=global_threshold
    elif threshold_modifier == TM_ADAPTIVE:
        local_threshold = get_adaptive_threshold(threshold_method, 
                                                 image, global_threshold,
                                                 mask, object_fraction,
                                                 two_class_otsu,
                                                 use_weighted_variance,
                                                 assign_middle_to_foreground)
        local_threshold = local_threshold * threshold_correction_factor
    elif threshold_modifier == TM_PER_OBJECT:
        local_threshold = get_per_object_threshold(threshold_method, image,
                                                   global_threshold,
                                                   mask, labels,
                                                   threshold_range_min,
                                                   threshold_range_max,
                                                   object_fraction,
                                                   two_class_otsu,
                                                   use_weighted_variance,
                                                   assign_middle_to_foreground)
        local_threshold = local_threshold * threshold_correction_factor
    else:
        raise NotImplementedError("%s thresholding is not implemented"%(threshold_modifier))
    if isinstance(local_threshold, np.ndarray):
        #
        # Constrain thresholds to within .7 to 1.5 of the global threshold.
        #
        threshold_range_min = max(threshold_range_min, global_threshold * .7)
        threshold_range_max = min(threshold_range_max, global_threshold * 1.5)
        if not threshold_range_min is None:
            local_threshold[local_threshold < threshold_range_min] = \
                threshold_range_min
        if not threshold_range_max is None:
            local_threshold[local_threshold > threshold_range_max] = \
            threshold_range_max
        if threshold_modifier == TM_PER_OBJECT:
            local_threshold[labels == 0] = 1.0
    else:
        if not threshold_range_min is None:
            local_threshold = max(local_threshold, threshold_range_min)
        if not threshold_range_max is None:
            local_threshold = min(local_threshold, threshold_range_max)
    return local_threshold, global_threshold

def get_global_threshold(threshold_method, image, mask = None,
                         object_fraction = 0.2,
                         two_class_otsu = True,
                         use_weighted_variance = True,
                         assign_middle_to_foreground = True):
    """Compute a single threshold over the whole image"""
    if mask is not None and not np.any(mask):
        return 1
    
    if threshold_method == TM_OTSU:
        return get_otsu_threshold(image, mask,
                                  two_class_otsu,
                                  use_weighted_variance,
                                  assign_middle_to_foreground)
    elif threshold_method == TM_MOG:
        return get_mog_threshold(image, mask, object_fraction)
    elif threshold_method == TM_BACKGROUND:
        return get_background_threshold(image, mask)
    elif threshold_method == TM_ROBUST_BACKGROUND:
        return get_robust_background_threshold(image,mask)
    elif threshold_method == TM_RIDLER_CALVARD:
        return get_ridler_calvard_threshold(image, mask)
    elif threshold_method == TM_KAPUR:
        return get_kapur_threshold(image,mask)
    else:
        raise NotImplementedError("%s algorithm not implemented"%(threshold_method))

def get_adaptive_threshold(threshold_method, image, threshold,
                           mask = None,
                           object_fraction = 0.2,
                           two_class_otsu = True,
                           use_weighted_variance = True,
                           assign_middle_to_foreground = True):
    
    """Given a global threshold, compute a threshold per pixel
    
    Break the image into blocks, computing the threshold per block.
    Afterwards, constrain the block threshold to .7 T < t < 1.5 T.
    
    Block sizes must be at least 50x50. Images > 500 x 500 get 10x10
    blocks.
    """
    # for the X and Y direction, find the # of blocks, given the
    # size constraints
    image_size = np.array(image.shape[:2],dtype=int)
    block_size = image_size / 10
    block_size[block_size<50] = 50
    nblocks = image_size / block_size
    #
    # Use a floating point block size to apportion the roundoff
    # roughly equally to each block
    #
    increment = ( np.array(image_size,dtype=float) / 
                  np.array(nblocks,dtype=float))
    #
    # Put the answer here
    #
    thresh_out = np.zeros(image_size, image.dtype)
    #
    # Loop once per block, computing the "global" threshold within the
    # block.
    #
    for i in range(nblocks[0]):
        i0 = int(i*increment[0])
        i1 = int((i+1)*increment[0])
        for j in range(nblocks[1]):
            j0 = int(j*increment[1])
            j1 = int((j+1)*increment[1])
            block = image[i0:i1,j0:j1]
            block_mask = None if mask is None else mask[i0:i1,j0:j1]
            block_threshold = get_global_threshold(threshold_method, 
                                                   block, mask = block_mask,
                                                   object_fraction = object_fraction,
                                                   two_class_otsu = two_class_otsu,
                                                   use_weighted_variance = use_weighted_variance,
                                                   assign_middle_to_foreground = assign_middle_to_foreground)
            thresh_out[i0:i1,j0:j1] = block_threshold
    return thresh_out

def get_per_object_threshold(method, image, threshold, mask=None, labels=None,
                             threshold_range_min = None,
                             threshold_range_max = None,
                             object_fraction = 0.2,
                             two_class_otsu = True,
                             use_weighted_variance = True,
                             assign_middle_to_foreground = True):
    """Return a matrix giving threshold per pixel calculated per-object
    
    image - image to be thresholded
    mask  - mask out "don't care" pixels
    labels - a label mask indicating object boundaries
    threshold - the global threshold
    """
    if labels is None:
        labels = np.ones(image.shape,int)
        if not mask is None:
            labels[np.logical_not(mask)] = 0 
    label_extents = scipy.ndimage.find_objects(labels,np.max(labels))
    local_threshold = np.ones(image.shape,image.dtype)
    for i,extent in zip(range(1,len(label_extents)+1),label_extents):
        label_mask = labels[extent]==i
        if not mask is None:
            label_mask = np.logical_and(mask[extent], label_mask)
        values = image[extent]
        per_object_threshold = get_global_threshold(method, values, 
                                                    mask = label_mask,
                                                    object_fraction = object_fraction,
                                                   two_class_otsu = two_class_otsu,
                                                   use_weighted_variance = use_weighted_variance,
                                                   assign_middle_to_foreground = assign_middle_to_foreground)
        local_threshold[extent][label_mask] = per_object_threshold
    return local_threshold

def get_otsu_threshold(image, mask = None, 
                       two_class_otsu = True,
                       use_weighted_variance = True,
                       assign_middle_to_foreground = True):
    if not mask is None:
        image = image[mask]
        if len(image) == 0:
            return 1
    else:
        image = np.array(image.flat)
    image, d = log_transform(image)
    if two_class_otsu:
        if use_weighted_variance:
            threshold = otsu(image)
        else:
            threshold = entropy(image)
    else:
        if use_weighted_variance:
            t1, t2 = otsu3(image)
        else:
            t1,t2 = entropy3(image)
        threshold = t1 if assign_middle_to_foreground else t2  
    threshold = inverse_log_transform(threshold, d)
    return threshold
        
def get_mog_threshold(image, mask=None, object_fraction = 0.2):
    """Compute a background using a mixture of gaussians
    
    This function finds a suitable
    threshold for the input image Block. It assumes that the pixels in the
    image belong to either a background class or an object class. 'pObject'
    is an initial guess of the prior probability of an object pixel, or
    equivalently, the fraction of the image that is covered by objects.
    Essentially, there are two steps. First, a number of Gaussian
    distributions are estimated to match the distribution of pixel
    intensities in OrigImage. Currently 3 Gaussian distributions are
    fitted, one corresponding to a background class, one corresponding to
    an object class, and one distribution for an intermediate class. The
    distributions are fitted using the Expectation-Maximization (EM)
    algorithm, a procedure referred to as Mixture of Gaussians modeling.
    When the 3 Gaussian distributions have been fitted, it's decided
    whether the intermediate class models background pixels or object
    pixels based on the probability of an object pixel 'pObject' given by
    the user.        
    """
    cropped_image = np.array(image.flat) if mask is None else image[mask]
    pixel_count = np.product(cropped_image.shape)
    max_count   = 512**2 # maximum # of pixels analyzed
    #
    # We need at least 3 pixels to keep from crashing because the highest 
    # and lowest are chopped out below.
    #
    object_fraction = float(object_fraction)
    background_fraction = 1.0-object_fraction
    if pixel_count < 3/min(object_fraction,background_fraction):
        return 1
    if np.max(cropped_image)==np.min(cropped_image):
        return cropped_image[0]
    number_of_classes = 3
    if pixel_count > max_count:
        np.random.seed(0)
        pixel_indices = np.random.permutation(pixel_count)[:max_count]
        cropped_image = cropped_image[pixel_indices]
    # Initialize mean and standard deviations of the three Gaussian
    # distributions by looking at the pixel intensities in the original
    # image and by considering the percentage of the image that is
    # covered by object pixels. Class 1 is the background class and Class
    # 3 is the object class. Class 2 is an intermediate class and we will
    # decide later if it encodes background or object pixels. Also, for
    # robustness the we remove 1% of the smallest and highest intensities
    # in case there are any quantization effects that have resulted in
    # unnaturally many 0:s or 1:s in the image.
    cropped_image.sort()
    one_percent = (np.product(cropped_image.shape) + 99)/100
    cropped_image=cropped_image[one_percent:-one_percent]
    pixel_count = np.product(cropped_image.shape)
    # Guess at the class means for the 3 classes: background,
    # in-between and object
    bg_pixel = cropped_image[round(pixel_count * background_fraction/2.0)]
    fg_pixel = cropped_image[round(pixel_count * (1-object_fraction/2))]
    class_mean = np.array([bg_pixel, (bg_pixel+fg_pixel)/2,fg_pixel])
    class_std = np.ones((3,)) * 0.15
    # Initialize prior probabilities of a pixel belonging to each class.
    # The intermediate class steals some probability from the background
    # and object classes.
    class_prob = np.array([3.0/4.0 * background_fraction ,
                              1.0/4.0,
                              3.0/4.0 * object_fraction])
    # Expectation-Maximization algorithm for fitting the three Gaussian
    # distributions/classes to the data. Note, the code below is general
    # and works for any number of classes. Iterate until parameters don't
    # change anymore.
    class_count = np.prod(class_mean.shape)
    #
    # Do a coarse iteration on subsampled data and a fine iteration on the real
    # data
    #
    r = np.random.RandomState()
    r.seed(cropped_image[:100].tolist())
    for data in (
        r.permutation(cropped_image)[0:(len(cropped_image) / 10)],
        cropped_image):
        delta = 1
        pixel_count = len(data)
        while delta > 0.001:
            old_class_mean = class_mean.copy()
            # Update probabilities of a pixel belonging to the background or
            # object1 or object2
            pixel_class_prob = np.ndarray((pixel_count,class_count))
            for k in range(class_count):
                norm = scipy.stats.norm(class_mean[k],class_std[k])
                pixel_class_prob[:,k] = class_prob[k] * norm.pdf(data)
            pixel_class_normalizer = np.sum(pixel_class_prob,1)+.000000000001
            for k in range(class_count):
                pixel_class_prob[:,k] = pixel_class_prob[:,k] / pixel_class_normalizer
                # Update parameters in Gaussian distributions
                class_prob[k] = np.mean(pixel_class_prob[:,k])
                class_mean[k] = (np.sum(pixel_class_prob[:,k] * data) /
                                 (class_prob[k] * pixel_count))
                class_std[k] = \
                    math.sqrt(np.sum(pixel_class_prob[:,k] * 
                                        (data-class_mean[k])**2)/
                              (pixel_count * class_prob[k])) + .000001
            delta = np.sum(np.abs(old_class_mean - class_mean))
    # Now the Gaussian distributions are fitted and we can describe the
    # histogram of the pixel intensities as the sum of these Gaussian
    # distributions. To find a threshold we first have to decide if the
    # intermediate class 2 encodes background or object pixels. This is
    # done by choosing the combination of class probabilities "class_prob"
    # that best matches the user input "object_fraction".
    
    # Construct an equally spaced array of values between the background
    # and object mean
    ndivisions = 10000
    level = (np.array(range(ndivisions)) *
             ((class_mean[2]-class_mean[0]) / ndivisions)
             + class_mean[0])
    class_gaussian = np.ndarray((ndivisions,class_count))
    for k in range(class_count):
        norm = scipy.stats.norm(class_mean[k],class_std[k])
        class_gaussian[:,k] = class_prob[k] * norm.pdf(level)
    if (abs(class_prob[1]+class_prob[2]-object_fraction) <
        abs(class_prob[2]-object_fraction)):
        # classifying the intermediate as object more closely models
        # the user's desired object fraction
        background_distribution = class_gaussian[:,0]
        object_distribution = class_gaussian[:,1]+class_gaussian[:,2]
    else:
        background_distribution = class_gaussian[:,0]+class_gaussian[:,1]
        object_distribution = class_gaussian[:,2]
    # Now, find the threshold at the intersection of the background
    # distribution and the object distribution.
    index = np.argmin(np.abs(background_distribution - object_distribution))
    return level[index]

def get_background_threshold(image, mask = None):
    """Get threshold based on the mode of the image
    The threshold is calculated by calculating the mode and multiplying by
    2 (an arbitrary empirical factor). The user will presumably adjust the
    multiplication factor as needed."""
    cropped_image = np.array(image.flat) if mask is None else image[mask]
    if np.product(cropped_image.shape)==0:
        return 0
    img_min = np.min(cropped_image)
    img_max = np.max(cropped_image)
    if img_min == img_max:
        return cropped_image[0]
    
    # Only do the histogram between values a bit removed from saturation
    robust_min = 0.02 * (img_max - img_min) + img_min
    robust_max = 0.98 * (img_max - img_min) + img_min
    nbins = 256
    cropped_image = cropped_image[np.logical_and(cropped_image > robust_min,
                                                 cropped_image < robust_max)]
    if len(cropped_image) == 0:
        return robust_min
    
    h = scipy.ndimage.histogram(cropped_image, robust_min, robust_max, nbins)
    index = np.argmax(h)
    cutoff = float(index) / float(nbins-1)
    #
    # If we have a low (or almost no) background, the cutoff will be
    # zero since the background falls into the lowest bin. We want to
    # offset by the robust cutoff factor of .02. We rescale by 1.04
    # to account for the 0.02 at the top and bottom.
    #
    cutoff = (cutoff + 0.02) / 1.04
    return img_min + cutoff * 2 * (img_max - img_min)

def get_robust_background_threshold(image, mask = None):
    """Calculate threshold based on mean & standard deviation
       The threshold is calculated by trimming the top and bottom 5% of
       pixels off the image, then calculating the mean and standard deviation
       of the remaining image. The threshold is then set at 2 (empirical
       value) standard deviations above the mean.""" 

    cropped_image = np.array(image.flat) if mask is None else image[mask]
    if np.product(cropped_image.shape)<3:
        return 0
    if np.min(cropped_image) == np.max(cropped_image):
        return cropped_image[0]
    
    cropped_image.sort()
    chop = int(round(np.product(cropped_image.shape) * .05))
    im   = cropped_image[chop:-chop]
    mean = im.mean()
    sd   = im.std()
    return mean+sd*2

def get_ridler_calvard_threshold(image, mask = None):
    """Find a threshold using the method of Ridler and Calvard
    
    The reference for this method is:
    "Picture Thresholding Using an Iterative Selection Method" 
    by T. Ridler and S. Calvard, in IEEE Transactions on Systems, Man and
    Cybernetics, vol. 8, no. 8, August 1978.
    """
    cropped_image = np.array(image.flat) if mask is None else image[mask]
    if np.product(cropped_image.shape)<3:
        return 0
    if np.min(cropped_image) == np.max(cropped_image):
        return cropped_image[0]
    
    # We want to limit the dynamic range of the image to 256. Otherwise,
    # an image with almost all values near zero can give a bad result.
    min_val = np.max(cropped_image)/256;
    cropped_image[cropped_image < min_val] = min_val;
    im = np.log(cropped_image);
    min_val = np.min(im);
    max_val = np.max(im);
    im = (im - min_val)/(max_val - min_val);
    pre_thresh = 0;
    # This method needs an initial value to start iterating. Using
    # graythresh (Otsu's method) is probably not the best, because the
    # Ridler Calvard threshold ends up being too close to this one and in
    # most cases has the same exact value.
    new_thresh = otsu(im)
    delta = 0.00001;
    while abs(pre_thresh - new_thresh)>delta:
        pre_thresh = new_thresh;
        mean1 = np.mean(im[im<pre_thresh]);
        mean2 = np.mean(im[im>=pre_thresh]);
        new_thresh = np.mean([mean1,mean2]);
    return math.exp(min_val + (max_val-min_val)*new_thresh);

def get_kapur_threshold(image, mask=None):
    """The Kapur, Sahoo, & Wong method of thresholding, adapted to log-space."""
    cropped_image = np.array(image.flat) if mask is None else image[mask]
    if np.product(cropped_image.shape)<3:
        return 0
    if np.min(cropped_image) == np.max(cropped_image):
        return cropped_image[0]
    log_image = np.log2(smooth_with_noise(cropped_image, 8))
    min_log_image = np.min(log_image)
    max_log_image = np.max(log_image)
    histogram = scipy.ndimage.histogram(log_image,
                                        min_log_image,
                                        max_log_image,
                                        256)
    histogram_values = (min_log_image + (max_log_image - min_log_image)*
                        np.array(range(256),float) / 255)
    # drop any zero bins
    keep = histogram != 0
    histogram = histogram[keep]
    histogram_values = histogram_values[keep]
    # check for corner cases
    if np.product(histogram_values)==1:
        return 2**histogram_values[0] 
    # Normalize to probabilities
    p = histogram.astype(float) / float(np.sum(histogram))
    # Find the probabilities totals up to and above each possible threshold.
    lo_sum = np.cumsum(p);
    hi_sum = lo_sum[-1] - lo_sum;
    lo_e = np.cumsum(p * np.log2(p));
    hi_e = lo_e[-1] - lo_e;

    # compute the entropies
    lo_entropy = lo_e / lo_sum - np.log2(lo_sum);
    hi_entropy = hi_e / hi_sum - np.log2(hi_sum);

    sum_entropy = lo_entropy[:-1] + hi_entropy[:-1];
    sum_entropy[np.logical_not(np.isfinite(sum_entropy))] = np.Inf
    entry = np.argmin(sum_entropy);
    return 2**((histogram_values[entry] + histogram_values[entry+1]) / 2);

def weighted_variance(image,mask,threshold):
    """Compute the log-transformed variance of foreground and background"""
    if not np.any(mask):
        return 0
    #
    # Clamp the dynamic range of the foreground
    #
    minval = np.max(image[mask])/256
    if minval == 0:
        return 0
    clamped_image = image[mask]
    clamped_image[clamped_image < minval] = minval
    
    if isinstance(threshold,np.ndarray):
        threshold = threshold[mask]
    fg = np.log2(clamped_image[clamped_image >=threshold])
    bg = np.log2(clamped_image[clamped_image < threshold])
    nfg = np.product(fg.shape)
    nbg = np.product(bg.shape)
    if nfg == 0:
        return np.var(bg)
    elif nbg == 0:
        return np.var(fg)
    else:
        return (np.var(fg) * nfg + np.var(bg)*nbg) / (nfg+nbg)

def sum_of_entropies(image, mask, threshold):
    """Bin the foreground and background pixels and compute the entropy 
    of the distribution of points among the bins
    """
    mask=mask.copy()
    mask[np.isnan(image)] = False
    if not np.any(mask):
        return 0
    #
    # Clamp the dynamic range of the foreground
    #
    minval = np.max(image[mask])/256
    if minval == 0:
        return 0
    clamped_image = image.copy()
    clamped_image[clamped_image < minval] = minval
    #
    # Smooth image with -8 bits of noise
    #
    image = smooth_with_noise(clamped_image, 8)
    im_min = np.min(image)
    im_max = np.max(image)
    #
    # Figure out the bounds for the histogram
    #
    upper = np.log2(im_max)
    lower = np.log2(im_min)
    if upper == lower:
        # All values are the same, answer is log2 of # of pixels
        return math.log(np.sum(mask),2) 
    #
    # Create log-transformed lists of points in the foreground and background
    # 
    fg = image[np.logical_and(mask, image >= threshold)]
    bg = image[np.logical_and(mask, image < threshold)]
    if len(fg) == 0 or len(bg) == 0:
        return 0
    log_fg = np.log2(fg)
    log_bg = np.log2(bg)
    #
    # Make these into histograms
    hfg = numpy_histogram(log_fg, 256, range=(lower,upper))[0]
    hbg = numpy_histogram(log_bg, 256, range=(lower,upper))[0]
    #hfg = scipy.ndimage.histogram(log_fg,lower,upper,256)
    #hbg = scipy.ndimage.histogram(log_bg,lower,upper,256)
    #
    # Drop empty bins
    #
    hfg = hfg[hfg>0]
    hbg = hbg[hbg>0]
    if np.product(hfg.shape) == 0:
        hfg = np.ones((1,),int)
    if np.product(hbg.shape) == 0:
        hbg = np.ones((1,),int)
    #
    # Normalize
    #
    hfg = hfg.astype(float) / float(np.sum(hfg))
    hbg = hbg.astype(float) / float(np.sum(hbg))
    #
    # Compute sum of entropies
    #
    return np.sum(hfg * np.log2(hfg)) + np.sum(hbg*np.log2(hbg))

def log_transform(image):
    '''Renormalize image intensities to log space
    
    Returns a tuple of transformed image and a dictionary to be passed into
    inverse_log_transform. The minimum and maximum from the dictionary
    can be applied to an image by the inverse_log_transform to 
    convert it back to its former intensity values.
    '''
    orig_min, orig_max = scipy.ndimage.extrema(image)[:2]
    #
    # We add 1/2 bit noise to an 8 bit image to give the log a bottom
    #
    limage = image.copy()
    noise_min = orig_min + (orig_max-orig_min)/256.0+np.finfo(image.dtype).eps
    limage[limage < noise_min] = noise_min
    d = { "noise_min":noise_min}
    limage = np.log(limage)
    log_min, log_max = scipy.ndimage.extrema(limage)[:2]
    d["log_min"] = log_min
    d["log_max"] = log_max
    return stretch(limage), d

def inverse_log_transform(image, d):
    '''Convert the values in image back to the scale prior to log_transform
    
    image - an image or value or values similarly scaled to image
    d - object returned by log_transform
    '''
    return np.exp(unstretch(image, d["log_min"], d["log_max"]))

def numpy_histogram(a, bins=10, range=None, normed=False, weights=None):
    '''A version of numpy.histogram that accounts for numpy's version'''
    args = inspect.getargs(np.histogram.func_code)[0]
    if args[-1] == "new":
        return np.histogram(a, bins, range, normed, weights, new=True)
    return np.histogram(a, bins, range, normed, weights)
    
