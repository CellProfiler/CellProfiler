'''filter.py - functions for applying filters to images

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import numpy as np
import _filter
from rankorder import rank_order
import scipy.ndimage as scind
from scipy.ndimage import map_coordinates, label
from scipy.ndimage import convolve, correlate1d, gaussian_filter
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import generate_binary_structure
from smooth import smooth_with_function_and_mask
from cpmorphology import fixup_scipy_ndimage_result as fix
from cpmorphology import centers_of_labels
from cpmorphology import grey_erosion, grey_reconstruction

def stretch(image, mask=None):
    '''Normalize an image to make the minimum zero and maximum one
    
    image - pixel data to be normalized
    mask  - optional mask of relevant pixels. None = don't mask
    
    returns the stretched image
    '''
    image = np.array(image, float)
    if np.product(image.shape) == 0:
        return image
    if mask is None:
        minval = np.min(image)
        maxval = np.max(image)
        if minval == maxval:
            if minval < 0:
                return np.zeros_like(image)
            elif minval > 1:
                return np.ones_like(image)
            return image
        else:
            return (image - minval) / (maxval - minval)
    else:
        significant_pixels = image[mask]
        if significant_pixels.size == 0:
            return image
        minval = np.min(significant_pixels)
        maxval = np.max(significant_pixels)
        if minval == maxval:
            transformed_image = minval
        else:
            transformed_image = ((significant_pixels - minval) /
                                 (maxval - minval))
        result = image.copy()
        image[mask] = transformed_image
        return image

def unstretch(image, minval, maxval):
    '''Perform the inverse of stretch, given a stretched image
    
    image - an image stretched by stretch or similarly scaled value or values
    minval - minimum of previously stretched image
    maxval - maximum of previously stretched image
    '''
    return image * (maxval - minval) + minval

def median_filter(data, mask, radius, percent=50):
    '''Masked median filter with octagonal shape
    
    data - array of data to be median filtered.
    mask - mask of significant pixels in data
    radius - the radius of a circle inscribed into the filtering octagon
    percent - conceptually, order the significant pixels in the octagon,
              count them and choose the pixel indexed by the percent
              times the count divided by 100. More simply, 50 = median
    returns a filtered array.  In areas where the median filter does
      not overlap the mask, the filtered result is undefined, but in
      practice, it will be the lowest value in the valid area.
    '''
    if np.all(~ mask):
        return data.copy()
    #
    # Normalize the ranked data to 0-255
    #
    if (not np.issubdtype(data.dtype, np.int) or
        np.min(data) < 0 or np.max(data) > 255):
        ranked_data,translation = rank_order(data[mask])
        max_ranked_data = np.max(ranked_data)
        if max_ranked_data == 0:
            return data
        if max_ranked_data > 255:
            ranked_data = ranked_data * 255 / max_ranked_data
        was_ranked = True
    else:
        ranked_data = data[mask]
        was_ranked = False
    input = np.zeros(data.shape, np.uint8 )
    input[mask] = ranked_data
    
    mmask = np.ascontiguousarray(mask, np.uint8)
    
    output = np.zeros(data.shape, np.uint8)
    
    _filter.median_filter(input, mmask, output, radius, percent)
    if was_ranked:
        #
        # The translation gives the original value at each ranking.
        # We rescale the output to the original ranking and then
        # use the translation to look up the original value in the data.
        #
        if max_ranked_data > 255:
            result = translation[output.astype(np.uint32) * max_ranked_data / 255]
        else:
            result = translation[output]
    else:
        result = output
    return result

def bilateral_filter(image, mask, sigma_spatial, sigma_range,
                     sampling_spatial = None, sampling_range = None):
    """Bilateral filter of an image
    
    image - image to be bilaterally filtered
    mask  - mask of significant points in image
    sigma_spatial - standard deviation of the spatial Gaussian
    sigma_range   - standard deviation of the range Gaussian
    sampling_spatial - amt to reduce image array extents when sampling
                       default is 1/2 sigma_spatial
    sampling_range - amt to reduce the range of values when sampling
                     default is 1/2 sigma_range
    
    The bilateral filter is described by the following equation:
    
    sum(Fs(||p - q||)Fr(|Ip - Iq|)Iq) / sum(Fs(||p-q||)Fr(|Ip - Iq))
    where the sum is over all points in the kernel
    p is all coordinates in the image
    q is the coordinates as perturbed by the mask
    Ip is the intensity at p
    Iq is the intensity at q
    Fs is the spatial convolution function, for us a Gaussian that
    falls off as the distance between falls off
    Fr is the "range" distance which falls off as the difference
    in intensity increases.
    
    1 / sum(Fs(||p-q||)Fr(|Ip - Iq)) is the weighting for point p
    
    """
    # The algorithm is taken largely from code by Jiawen Chen which miraculously
    # extends to the masked case:
    # http://groups.csail.mit.edu/graphics/bilagrid/bilagrid_web.pdf
    #
    # Form a 3-d array whose extent is reduced in the i,j directions
    # by the spatial sampling parameter and whose extent is reduced in the
    # z (image intensity) direction by the range sampling parameter.
    
    # Scatter each significant pixel in the image into the nearest downsampled
    # array address where the pixel's i,j coordinate gives the corresponding
    # i and j in the matrix and the intensity value gives the corresponding z
    # in the array.
    
    # Count the # of values entered into each 3-d array element to form a
    # weight.
    
    # Similarly convolve the downsampled value and weight arrays with a 3-d
    # Gaussian kernel whose i and j Gaussian is the sigma_spatial and whose
    # z is the sigma_range.
    #
    # Divide the value by the weight to scale each z value appropriately
    #
    # Linearly interpolate using an i x j x 3 array where [:,:,0] is the
    # i coordinate in the downsampled array, [:,:,1] is the j coordinate
    # and [:,:,2] is the unrounded index of the z-slot 
    #
    # One difference is that I don't pad the intermediate arrays. The
    # weights bleed off the edges of the intermediate arrays and this
    # accounts for the ring of zero values used at the border bleeding
    # back into the intermediate arrays during convolution
    #
     
    if sampling_spatial is None:
        sampling_spatial = sigma_spatial / 2.0
    if sampling_range is None:
        sampling_range = sigma_range / 2.0
    
    if np.all(np.logical_not(mask)):
        return image
    masked_image = image[mask]
    image_min = np.min(masked_image)
    image_max = np.max(masked_image)
    image_delta = image_max - image_min
    if image_delta == 0:
        return image
    
    #
    # ds = downsampled. Calculate the ds array sizes and sigmas.
    #
    ds_sigma_spatial = sigma_spatial / sampling_spatial
    ds_sigma_range   = sigma_range / sampling_range
    ds_i_limit       = int(image.shape[0] / sampling_spatial) + 2
    ds_j_limit       = int(image.shape[1] / sampling_spatial) + 2
    ds_z_limit       = int(image_delta / sampling_range) + 2
    
    grid_data    = np.zeros((ds_i_limit, ds_j_limit, ds_z_limit))
    grid_weights = np.zeros((ds_i_limit, ds_j_limit, ds_z_limit))
    #
    # Compute the downsampled i, j and z coordinates at each point
    #
    di,dj = np.mgrid[0:image.shape[0],
                     0:image.shape[1]].astype(float) / sampling_spatial
    dz = (masked_image - image_min) / sampling_range
    #
    # Treat this as a list of 3-d coordinates from now on
    #
    di = di[mask]
    dj = dj[mask]
    #
    # scatter the unmasked image points into the data array and
    # scatter a value of 1 per point into the weights
    #
    grid_data[(di + .5).astype(int), 
              (dj + .5).astype(int),
              (dz + .5).astype(int)] += masked_image
    grid_weights[(di + .5).astype(int), 
                 (dj + .5).astype(int),
                 (dz + .5).astype(int)] += 1
    #
    # Make a Gaussian kernel
    #
    kernel_spatial_limit = int(2 * ds_sigma_spatial) + 1
    kernel_range_limit   = int(2 * ds_sigma_range) + 1
    ki,kj,kz = np.mgrid[-kernel_spatial_limit : kernel_spatial_limit+1,
                        -kernel_spatial_limit : kernel_spatial_limit+1,
                        -kernel_range_limit : kernel_range_limit+1]
    kernel = np.exp(-.5 * ((ki**2 + kj**2) / ds_sigma_spatial ** 2 +
                           kz**2 / ds_sigma_range ** 2))
     
    blurred_grid_data = convolve(grid_data, kernel, mode='constant')
    blurred_weights = convolve(grid_weights, kernel, mode='constant')
    weight_mask = blurred_weights > 0
    normalized_blurred_grid = np.zeros(grid_data.shape)
    normalized_blurred_grid[weight_mask] = ( blurred_grid_data[weight_mask] /
                                             blurred_weights[weight_mask])
    #
    # Now use di, dj and dz to find the coordinate of the point within
    # the blurred grid to use. We actually interpolate between points
    # here (both in the i,j direction to get intermediate z values and in
    # the z direction to get the slot, roughly where we put our original value)
    #
    dijz = np.vstack((di, dj, dz))
    image_copy = image.copy()
    image_copy[mask] = map_coordinates(normalized_blurred_grid, dijz,
                                       order = 1)
    return image_copy

def laplacian_of_gaussian(image, mask, size, sigma):
    '''Perform the Laplacian of Gaussian transform on the image
    
    image - 2-d image array
    mask  - binary mask of significant pixels
    size  - length of side of square kernel to use
    sigma - standard deviation of the Gaussian
    '''
    half_size = size/2
    i,j = np.mgrid[-half_size:half_size+1, 
                   -half_size:half_size+1].astype(float) / float(sigma)
    distance = (i**2 + j**2)/2
    gaussian = np.exp(-distance)
    #
    # Normalize the Gaussian
    #
    gaussian = gaussian / np.sum(gaussian)

    log = (distance - 1) * gaussian
    #
    # Normalize the kernel to have a sum of zero
    #
    log = log - np.mean(log)
    output = convolve(image, log, mode='constant', cval=100.0)
    if mask is None:
        mask = np.ones(image.shape,np.uint8)
    else:
        mask = np.array(mask, np.uint8)
    output = masked_convolution(image, mask, log)
    output[mask==0] = image[mask==0]
    return output

def masked_convolution(data, mask, kernel):
    data = np.ascontiguousarray(data, np.float64)
    kernel = np.ascontiguousarray(kernel, np.float64)
    return _filter.masked_convolution(data, mask, kernel)

def canny(image, mask, sigma, low_threshold, high_threshold):
    '''Edge filter an image using the Canny algorithm.
    
    sigma - the standard deviation of the Gaussian used
    low_threshold - threshold for edges that connect to high-threshold
                    edges
    high_threshold - threshold of a high-threshold edge
    
    Canny, J., A Computational Approach To Edge Detection, IEEE Trans. 
    Pattern Analysis and Machine Intelligence, 8:679-714, 1986
    
    William Green's Canny tutorial
    http://www.pages.drexel.edu/~weg22/can_tut.html
    '''
    #
    # The steps involved:
    #
    # * Smooth using the Gaussian with sigma above.
    #
    # * Apply the horizontal and vertical Sobel operators to get the gradients
    #   within the image. The edge strength is the sum of the magnitudes
    #   of the gradients in each direction.
    #
    # * Find the normal to the edge at each point using the arctangent of the
    #   ratio of the Y sobel over the X sobel - pragmatically, we can
    #   look at the signs of X and Y and the relative magnitude of X vs Y
    #   to sort the points into 4 categories: horizontal, vertical,
    #   diagonal and antidiagonal.
    #
    # * Look in the normal and reverse directions to see if the values
    #   in either of those directions are greater than the point in question.
    #   Use interpolation to get a mix of points instead of picking the one
    #   that's the closest to the normal.
    #
    # * Label all points above the high threshold as edges.
    # * Recursively label any point above the low threshold that is 8-connected
    #   to a labeled point as an edge.
    #
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?
    #
    fsmooth = lambda x: gaussian_filter(x, sigma, mode='constant')
    smoothed = smooth_with_function_and_mask(image, fsmooth, mask)
    jsobel = convolve(smoothed, [[-1,0,1],[-2,0,2],[-1,0,1]])
    isobel = convolve(smoothed, [[-1,-2,-1],[0,0,0],[1,2,1]])
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)
    magnitude = np.sqrt(isobel*isobel + jsobel*jsobel)
    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #
    s = generate_binary_structure(2,2)
    emask = binary_erosion(mask, s, border_value = 0)
    emask = np.logical_and(emask, magnitude > 0)
    #
    #--------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(image.shape,bool)
    #----- 0 to 45 degrees ------
    pts_plus = np.logical_and(isobel >= 0, 
                              np.logical_and(jsobel >= 0, 
                                             abs_isobel >= abs_jsobel))
    pts_minus = np.logical_and(isobel <= 0,
                               np.logical_and(jsobel <= 0,
                                              abs_isobel >= abs_jsobel))
    pts = np.logical_or(pts_plus, pts_minus)
    pts = np.logical_and(emask, pts)
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[1:,:][pts[:-1,:]]
    c2 = magnitude[1:,1:][pts[:-1,:-1]]
    m  = magnitude[pts]
    w  = abs_jsobel[pts] / abs_isobel[pts]
    c_plus  = c2 * w + c1 * (1-w) <= m
    c1 = magnitude[:-1,:][pts[1:,:]]
    c2 = magnitude[:-1,:-1][pts[1:,1:]]
    c_minus =  c2 * w + c1 * (1-w) <= m
    local_maxima[pts] = np.logical_and(c_plus, c_minus)
    #----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts_plus = np.logical_and(isobel >= 0, 
                              np.logical_and(jsobel >= 0, 
                                             abs_isobel <= abs_jsobel))
    pts_minus = np.logical_and(isobel <= 0,
                               np.logical_and(jsobel <= 0, 
                                              abs_isobel <= abs_jsobel))
    pts = np.logical_or(pts_plus, pts_minus)
    pts = np.logical_and(emask, pts)
    c1 = magnitude[:,1:][pts[:,:-1]]
    c2 = magnitude[1:,1:][pts[:-1,:-1]]
    m  = magnitude[pts]
    w  = abs_isobel[pts] / abs_jsobel[pts]
    c_plus  = c2 * w + c1 * (1-w) <= m
    c1 = magnitude[:,:-1][pts[:,1:]]
    c2 = magnitude[:-1,:-1][pts[1:,1:]]
    c_minus =  c2 * w + c1 * (1-w) <= m
    local_maxima[pts] = np.logical_and(c_plus, c_minus)
    #----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = np.logical_and(isobel <= 0, 
                              np.logical_and(jsobel >= 0, 
                                             abs_isobel <= abs_jsobel))
    pts_minus = np.logical_and(isobel >= 0,
                               np.logical_and(jsobel <= 0, 
                                              abs_isobel <= abs_jsobel))
    pts = np.logical_or(pts_plus, pts_minus)
    pts = np.logical_and(emask, pts)
    c1a = magnitude[:,1:][pts[:,:-1]]
    c2a = magnitude[:-1,1:][pts[1:,:-1]]
    m  = magnitude[pts]
    w  = abs_isobel[pts] / abs_jsobel[pts]
    c_plus  = c2a * w + c1a * (1.0-w) <= m
    c1 = magnitude[:,:-1][pts[:,1:]]
    c2 = magnitude[1:,:-1][pts[:-1,1:]]
    c_minus =  c2 * w + c1 * (1.0-w) <= m
    cc = np.logical_and(c_plus,c_minus)
    local_maxima[pts] = np.logical_and(c_plus, c_minus)
    #----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts_plus = np.logical_and(isobel <= 0, 
                              np.logical_and(jsobel >= 0, 
                                             abs_isobel >= abs_jsobel))
    pts_minus = np.logical_and(isobel >= 0,
                               np.logical_and(jsobel <= 0, 
                                              abs_isobel >= abs_jsobel))
    pts = np.logical_or(pts_plus, pts_minus)
    pts = np.logical_and(emask, pts)
    c1 = magnitude[:-1,:][pts[1:,:]]
    c2 = magnitude[:-1,1:][pts[1:,:-1]]
    m  = magnitude[pts]
    w  = abs_jsobel[pts] / abs_isobel[pts]
    c_plus  = c2 * w + c1 * (1-w) <= m
    c1 = magnitude[1:,:][pts[:-1,:]]
    c2 = magnitude[1:,:-1][pts[:-1,1:]]
    c_minus =  c2 * w + c1 * (1-w) <= m
    local_maxima[pts] = np.logical_and(c_plus, c_minus)
    #
    #---- Create two masks at the two thresholds.
    #
    high_mask = np.logical_and(local_maxima, magnitude >= high_threshold)
    low_mask  = np.logical_and(local_maxima, magnitude >= low_threshold)
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them 
    #
    labels,count = label(low_mask, np.ndarray((3,3),bool))
    if count == 0:
        return low_mask
    
    sums = fix(scind.sum(high_mask, labels, np.arange(count)+1))
    good_label = np.zeros((count+1,),bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    return output_mask  

def roberts(image, mask=None):
    '''Find edges using the Roberts algorithm
    
    image - the image to process
    mask  - mask of relevant points
    
    The algorithm returns the magnitude of the output of the two Roberts
    convolution kernels.
    
    The following is the canonical citation for the algorithm:
    L. Roberts Machine Perception of 3-D Solids, Optical and 
    Electro-optical Information Processing, MIT Press 1965.
    
    The following website has a tutorial on the algorithm:
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/roberts.htm
    '''
    result = np.zeros(image.shape)
    #
    # Four quadrants and two convolutions:
    #
    # q0,0 | q0,1    1  |  0  anti-diagonal
    # q1,0 | q1,1    0  | -1
    #
    # q-1,0 | q0,0   0  |  1  diagonal
    # q-1,1 | q0,1  -1  |  0
    #
    # Points near the mask edges and image edges are computed unreliably
    # so make them zero (no edge) in the result
    #
    if mask == None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2,2),
                              border_value = 0)
    result[big_mask==False] = 0
    q00   = image[:,:][big_mask]
    q11   = image[1:,1:][big_mask[:-1,:-1]]
    qm11  = image[:-1,1:][big_mask[1:,:-1]]
    diagonal = q00 - qm11
    anti_diagonal = q00 - q11
    result[big_mask] = np.sqrt(diagonal*diagonal + anti_diagonal*anti_diagonal)
    return result

def sobel(image, mask=None):
    '''Calculate the absolute magnitude Sobel to find the edges
    
    image - image to process
    mask - mask of relevant points
    
    Take the square root of the sum of the squares of the horizontal and
    vertical Sobels to get a magnitude that's somewhat insensitive to
    direction.
    
    Note that scipy's Sobel returns a directional Sobel which isn't
    useful for edge detection in its raw form.
    '''
    return np.sqrt(hsobel(image,mask)**2 + vsobel(image,mask)**2)

def hsobel(image, mask=None):
    '''Find the horizontal edges of an image using the Sobel transform
    
    image - image to process
    mask  - mask of relevant points
    
    We use the following kernel and return the absolute value of the
    result at each point:
     1   2   1
     0   0   0
    -1  -2  -1
    '''
    if mask == None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2,2),
                              border_value = 0)
    result = np.abs(convolve(image, np.array([[ 1, 2, 1],
                                              [ 0, 0, 0],
                                              [-1,-2,-1]]).astype(float)/4.0))
    result[big_mask==False] = 0
    return result

def vsobel(image, mask=None):
    '''Find the vertical edges of an image using the Sobel transform
    
    image - image to process
    mask  - mask of relevant points
    
    We use the following kernel and return the absolute value of the
    result at each point:
     1   0  -1
     2   0  -2
     1   0  -1
    '''
    if mask == None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2,2),
                              border_value = 0)
    result = np.abs(convolve(image, np.array([[ 1, 0,-1],
                                              [ 2, 0,-2],
                                              [ 1, 0,-1]]).astype(float)/4.0))
    result[big_mask==False] = 0
    return result

def prewitt(image, mask=None):
    '''Find the edge magnitude using the Prewitt transform
    
    image - image to process
    mask  - mask of relevant points
    
    Return the square root of the sum of squares of the horizontal
    and vertical Prewitt transforms.
    '''
    return np.sqrt(hprewitt(image,mask)**2 + vprewitt(image,mask)**2)
     
def hprewitt(image, mask=None):
    '''Find the horizontal edges of an image using the Prewitt transform
    
    image - image to process
    mask  - mask of relevant points
    
    We use the following kernel and return the absolute value of the
    result at each point:
     1   1   1
     0   0   0
    -1  -1  -1
    '''
    if mask == None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2,2),
                              border_value = 0)
    result = np.abs(convolve(image, np.array([[ 1, 1, 1],
                                              [ 0, 0, 0],
                                              [-1,-1,-1]]).astype(float)/3.0))
    result[big_mask==False] = 0
    return result

def vprewitt(image, mask=None):
    '''Find the vertical edges of an image using the Prewitt transform
    
    image - image to process
    mask  - mask of relevant points
    
    We use the following kernel and return the absolute value of the
    result at each point:
     1   0  -1
     1   0  -1
     1   0  -1
    '''
    if mask == None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2,2),
                              border_value = 0)
    result = np.abs(convolve(image, np.array([[ 1, 0,-1],
                                              [ 1, 0,-1],
                                              [ 1, 0,-1]]).astype(float)/3.0))
    result[big_mask==False] = 0
    return result

def gabor(image, labels, frequency, theta):
    '''Gabor-filter the objects in an image
    
    image - 2-d grayscale image to filter
    labels - a similarly shaped labels matrix
    frequency - cycles per trip around the circle
    theta - angle of the filter. 0 to 2 pi
    
    Calculate the Gabor filter centered on the centroids of each object
    in the image. Summing the resulting image over the labels matrix will
    yield a texture measure per object.
    '''
    #
    # The code inscribes the X and Y position of each pixel relative to
    # the centroid of that pixel's object. After that, the Gabor filter
    # for the image can be calculated per-pixel and the image can be
    # multiplied by the filter to get the filtered image.
    #
    nobjects = np.max(labels)
    if nobjects == 0:
        return image
    centers = centers_of_labels(labels)
    areas = fix(scind.sum(np.ones(image.shape),labels, np.arange(nobjects)+1))
    mask = labels > 0
    i,j = np.mgrid[0:image.shape[0],0:image.shape[1]].astype(float)
    i = i[mask]
    j = j[mask]
    image = image[mask]
    lm = labels[mask] - 1
    i -= centers[0,lm]
    j -= centers[1,lm]
    sigma = np.sqrt(areas/np.pi) / 3.0
    sigma = sigma[lm]
    g_exp = 1000.0/(2.0*np.pi*sigma**2) * np.exp(-(i**2 + j**2)/(2*sigma**2))
    g_angle = 2*np.pi/frequency*(i*np.cos(theta)+j*np.sin(theta))
    g_cos = g_exp * np.cos(g_angle)
    g_sin = g_exp * np.sin(g_angle)
    #
    # Normalize so that the sum of the filter over each object is zero
    # and so that there is no bias-value within each object.
    #
    g_cos_mean = fix(scind.mean(g_cos,lm, np.arange(nobjects)))
    i_mean = fix(scind.mean(image, lm, np.arange(nobjects)))
    i_norm = image - i_mean[lm]
    g_sin_mean = fix(scind.mean(g_sin,lm, np.arange(nobjects)))
    g_cos -= g_cos_mean[lm]
    g_sin -= g_sin_mean[lm]
    g = np.zeros(mask.shape,dtype=np.complex)
    g[mask] = i_norm *g_cos+i_norm * g_sin*1j
    return g

def enhance_dark_holes(image, min_radius, max_radius, mask=None):
    '''Enhance dark holes using a rolling ball filter

    image - grayscale 2-d image
    radii - a vector of radii: we enhance holes at each given radius
    '''
    #
    # Do 4-connected erosion
    #
    se = np.array([[False, True, False],
                   [True, True, True],
                   [False, True, False]])
    #
    # Invert the intensities
    #
    inverted_image = image.max() - image
    previous_reconstructed_image = inverted_image
    eroded_image = inverted_image
    smoothed_image = np.zeros(image.shape)
    for i in range(max_radius+1):
        eroded_image = grey_erosion(eroded_image, mask=mask, footprint = se)
        reconstructed_image = grey_reconstruction(eroded_image, inverted_image,
                                                  footprint = se)
        output_image = previous_reconstructed_image - reconstructed_image
        if i >= min_radius:
            smoothed_image += output_image
        previous_reconstructed_image = reconstructed_image
    smoothed_image[smoothed_image > 1] = 1
    smoothed_image[smoothed_image < 0] = 0
    return smoothed_image

def circular_average_filter(image, radius, mask=None):
    '''Blur an image using a circular averaging filter (pillbox)  

    image - grayscale 2-d image
    radii - radius of filter in pixels
    
    The filter will be within a square matrix of side 2*radius+1
    
    This code is translated straight from MATLAB's fspecial function
    '''
    
    crad = np.ceil(radius-0.5)
    x,y = np.mgrid[-crad:crad+1,-crad:crad+1].astype(float) 
    maxxy = np.maximum(abs(x),abs(y))
    minxy = np.minimum(abs(x),abs(y))
    
    m1 = ((radius **2 < (maxxy+0.5)**2 + (minxy-0.5)**2)*(minxy-0.5) + 
      (radius**2 >= (maxxy+0.5)**2 + (minxy-0.5)**2) * 
      np.real(np.sqrt(np.asarray(radius**2 - (maxxy + 0.5)**2,dtype=complex)))) 
    m2 = ((radius**2 >  (maxxy-0.5)**2 + (minxy+0.5)**2)*(minxy+0.5) + 
      (radius**2 <= (maxxy-0.5)**2 + (minxy+0.5)**2)*
      np.real(np.sqrt(np.asarray(radius**2 - (maxxy - 0.5)**2,dtype=complex))))
    
    sgrid = ((radius**2*(0.5*(np.arcsin(m2/radius) - np.arcsin(m1/radius)) + 
          0.25*(np.sin(2*np.arcsin(m2/radius)) - np.sin(2*np.arcsin(m1/radius)))) - 
         (maxxy-0.5)*(m2-m1) + (m1-minxy+0.5)) *  
         ((((radius**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) & 
         (radius**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) | 
         ((minxy == 0) & (maxxy-0.5 < radius) & (maxxy+0.5 >= radius)))) ) 
    
    sgrid = sgrid + ((maxxy+0.5)**2 + (minxy+0.5)**2 < radius**2) 
    sgrid[crad,crad] = np.minimum(np.pi*radius**2,np.pi/2) 
    if ((crad>0) and (radius > crad-0.5) and (radius**2 < (crad-0.5)**2+0.25)): 
        m1  = np.sqrt(radius**2 - (crad - 0.5)**2) 
        m1n = m1/radius 
        sg0 = 2*(radius**2*(0.5*np.arcsin(m1n) + 0.25*np.sin(2*np.arcsin(m1n)))-m1*(crad-0.5))
        sgrid[2*crad,crad]   = sg0
        sgrid[crad,2*crad]   = sg0
        sgrid[crad,0]        = sg0 
        sgrid[0,crad]        = sg0
        sgrid[2*crad-1,crad] = sgrid[2*crad-1,crad] - sg0
        sgrid[crad,2*crad-1] = sgrid[crad,2*crad-1] - sg0
        sgrid[crad,1]        = sgrid[crad,1]        - sg0 
        sgrid[1,crad]        = sgrid[1,crad]        - sg0 
    
    sgrid[crad,crad] = np.minimum(sgrid[crad,crad],1) 
    kernel = sgrid/sgrid.sum()
    
    output = convolve(image, kernel, mode='constant')
    if mask is None:
        mask = np.ones(image.shape,np.uint8)
    else:
        mask = np.array(mask, np.uint8)
    output = masked_convolution(image, mask, kernel)
    output[mask==0] = image[mask==0]
    
    return output
