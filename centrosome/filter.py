from __future__ import absolute_import
from __future__ import division
import deprecation
import numpy as np
import scipy.ndimage as scind
from scipy.ndimage import (binary_erosion, convolve, gaussian_filter,
                           generate_binary_structure, label,
                           map_coordinates)

from . import _filter
from .rankorder import rank_order
from .smooth import smooth_with_function_and_mask
from .cpmorphology import fixup_scipy_ndimage_result as fix
from .cpmorphology import (centers_of_labels, convex_hull_ijv,
                           get_line_pts, grey_dilation,
                           grey_erosion, grey_reconstruction)
from six.moves import range

'''# of points handled in the first pass of the convex hull code'''
CONVEX_HULL_CHUNKSIZE = 250000

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
    if mask is None:
        mask = np.ones(data.shape, dtype=bool)
    if np.all(~ mask):
        return data.copy()
    #
    # Normalize the ranked data to 0-255
    #
    if (not np.issubdtype(data.dtype, np.int) or
        np.min(data) < 0 or np.max(data) > 255):
        ranked_data, translation = rank_order(data[mask], nbins=255)
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
        result = translation[output]
    else:
        result = output
    return result


@deprecation.deprecated(
    current_version="2.0.0",
    deprecated_in="2.0.0",
    details="replaced by the `skimage.restoration.denoise_bilateral` function from `scikit-image` instead",
    removed_in="2.1.0"
)
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
    half_size = size//2
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
    if mask is None:
        mask = np.ones(image.shape[:2], bool)
    masked_image = image.copy()
    masked_image[~mask] = 0
    output = convolve(masked_image, log, mode='constant', cval=0)
    #
    # Do the LoG of the inverse of the mask. This finds the magnitude of the
    # contribution of the masked pixels. We then fudge by multiplying by the
    # value at the pixel of interest - this effectively sets the value at a
    # masked pixel to that of the pixel of interest.
    #
    # It underestimates the LoG, that's not a terrible thing.
    #
    correction = convolve((~ mask).astype(float), log, mode='constant', cval = 1)
    output += correction * image
    output[~ mask] = image[~ mask]
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
    labels,count = label(low_mask, np.ones((3,3),bool))
    if count == 0:
        return low_mask

    sums = fix(scind.sum(high_mask, labels, np.arange(count,dtype=np.int32)+1))
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
    if mask is None:
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
    if mask is None:
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
    if mask is None:
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
    if mask is None:
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
    if mask is None:
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
    areas = fix(scind.sum(np.ones(image.shape),labels, np.arange(nobjects, dtype=np.int32)+1))
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
            smoothed_image = np.maximum(smoothed_image,output_image)
        previous_reconstructed_image = reconstructed_image
    return smoothed_image


def granulometry_filter(image, min_radius, max_radius, mask=None):
    '''Enhances bright structures within a min and max radius using a rolling ball filter

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
    # Initialize
    #
    inverted_image = image.max() - image
    previous_opened_image = image
    eroded_image = image
    selected_granules_image = np.zeros(image.shape)
    #
    # Select granules by successive morphological openings
    #
    for i in range(max_radius+1):
        eroded_image = grey_erosion(eroded_image, mask=mask, footprint = se)
        opened_image = grey_dilation(eroded_image, inverted_image, footprint = se)
        output_image = previous_opened_image - opened_image
        if i >= min_radius:
            selected_granules_image = np.maximum(selected_granules_image, output_image)
        previous_opened_image = opened_image
    return selected_granules_image

def circular_average_filter(image, radius, mask=None):
    '''Blur an image using a circular averaging filter (pillbox)

    image - grayscale 2-d image
    radii - radius of filter in pixels

    The filter will be within a square matrix of side 2*radius+1

    This code is translated straight from MATLAB's fspecial function
    '''

    crad = int(np.ceil(radius-0.5))
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

#######################################
#
# Structure and ideas for the Kalman filter derived from u-track
# as described in
#
#  Jaqaman, "Robust single-particle tracking in live-cell
#  time-lapse sequences", NATURE METHODS | VOL.5 NO.8 | AUGUST 2008
#
#######################################
class KalmanState(object):
    '''A data structure representing the state at a frame

    The original method uses "feature" to mean the same thing as
    CellProfiler's "object".

    The state vector is somewhat abstract: it's up to the caller to
    determine what each of the indices mean. For instance, in a model
    with a 2-d position and velocity component, the state might be
    i, j, di, dj

    .observation_matrix - matrix to transform the state vector into the
                          observation vector. The observation matrix gives
                          the dimensions of the observation vector from its
                          i-shape and the dimensions of the state vector
                          from its j-shape. For observations of position
                          and states with velocity, the observation matrix
                          might be:

                          np.array([[1,0,0,0],
                                    [0,1,0,0]])

    .translation_matrix - matrix to translate the state vector from t-1 to t
                          For instance, the translation matrix for position
                          and velocity might be:

                          np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]])

    .state_vec - an array of vectors per feature

    .state_cov - the covariance matrix yielding the prediction. Each feature
                has a 4x4 matrix that can be used to predict the new value

    .noise_var - the variance of the state noise for each feature for each
                vector element

    .state_noise - a N x 4 array: the state noise for the i, j, vi and vj

    .state_noise_idx - the feature indexes for each state noise vector

    .obs_vec   - the prediction for the observed variables
    '''

    def __init__(self,
                 observation_matrix,
                 translation_matrix,
                 state_vec = None,
                 state_cov = None,
                 noise_var = None,
                 state_noise = None,
                 state_noise_idx = None):
        self.observation_matrix = observation_matrix
        self.translation_matrix = translation_matrix
        if state_vec is not None:
            self.state_vec = state_vec
        else:
            self.state_vec = np.zeros((0, self.state_len))
        if state_cov is not None:
            self.state_cov = state_cov
        else:
            self.state_cov = np.zeros((0, self.state_len, self.state_len))
        if noise_var is not None:
            self.noise_var = noise_var
        else:
            self.noise_var = np.zeros((0, self.state_len))
        if state_noise is not None:
            self.state_noise = state_noise
        else:
            self.state_noise = np.zeros((0, self.state_len))
        if state_noise_idx is not None:
            self.state_noise_idx = state_noise_idx
        else:
            self.state_noise_idx = np.zeros(0, int)

    @property
    def state_len(self):
        '''# of elements in the state vector'''
        return self.observation_matrix.shape[1]
    @property
    def obs_len(self):
        '''# of elements in the observation vector'''
        return self.observation_matrix.shape[0]

    @property
    def has_cached_predicted_state_vec(self):
        '''True if next state vec has been calculated'''
        return hasattr(self, "p_state_vec")

    @property
    def predicted_state_vec(self):
        '''The predicted state vector for the next time point

        From Welch eqn 1.9
        '''
        if not self.has_cached_predicted_state_vec:
            self.p_state_vec = dot_n(
                self.translation_matrix,
                self.state_vec[:, :, np.newaxis])[:,:,0]
        return self.p_state_vec

    @property
    def has_cached_obs_vec(self):
        '''True if the observation vector for the next state has been calculated'''
        return hasattr(self, "obs_vec")

    @property
    def predicted_obs_vec(self):
        '''The predicted observation vector

        The observation vector for the next step in the filter.
        '''
        if not self.has_cached_obs_vec:
            self.obs_vec = dot_n(
                self.observation_matrix,
                self.predicted_state_vec[:,:,np.newaxis])[:,:,0]
        return self.obs_vec

    def map_frames(self, old_indices):
        '''Rewrite the feature indexes based on the next frame's identities

        old_indices - for each feature in the new frame, the index of the
                      old feature
        '''
        nfeatures = len(old_indices)
        noldfeatures = len(self.state_vec)
        if nfeatures > 0:
            self.state_vec = self.state_vec[old_indices]
            self.state_cov = self.state_cov[old_indices]
            self.noise_var = self.noise_var[old_indices]
            if self.has_cached_obs_vec:
                self.obs_vec = self.obs_vec[old_indices]
            if self.has_cached_predicted_state_vec:
                self.p_state_vec = self.p_state_vec[old_indices]
            if len(self.state_noise_idx) > 0:
                #
                # We have to renumber the new_state_noise indices and get rid
                # of those that don't map to numbers. Typical index trick here:
                # * create an array for each legal old element: -1 = no match
                # * give each old element in the array the new number
                # * Filter out the "no match" elements.
                #
                reverse_indices = -np.ones(noldfeatures, int)
                reverse_indices[old_indices] = np.arange(nfeatures)
                self.state_noise_idx = reverse_indices[self.state_noise_idx]
                self.state_noise = self.state_noise[self.state_noise_idx != -1,:]
                self.state_noise_idx = self.state_noise_idx[self.state_noise_idx != -1]

    def add_features(self, kept_indices, new_indices,
                     new_state_vec, new_state_cov, new_noise_var):
        '''Add new features to the state

        kept_indices - the mapping from all indices in the state to new
                       indices in the new version

        new_indices - the indices of the new features in the new version

        new_state_vec - the state vectors for the new indices

        new_state_cov - the covariance matrices for the new indices

        new_noise_var - the noise variances for the new indices
        '''
        assert len(kept_indices) == len(self.state_vec)
        assert len(new_indices) == len(new_state_vec)
        assert len(new_indices) == len(new_state_cov)
        assert len(new_indices) == len(new_noise_var)

        if self.has_cached_obs_vec:
            del self.obs_vec
        if self.has_cached_predicted_state_vec:
            del self.predicted_obs_vec

        nfeatures = len(kept_indices) + len(new_indices)
        next_state_vec = np.zeros((nfeatures, self.state_len))
        next_state_cov = np.zeros((nfeatures, self.state_len, self.state_len))
        next_noise_var = np.zeros((nfeatures, self.state_len))

        if len(kept_indices) > 0:
            next_state_vec[kept_indices] = self.state_vec
            next_state_cov[kept_indices] = self.state_cov
            next_noise_var[kept_indices] = self.noise_var
            if len(self.state_noise_idx) > 0:
                self.state_noise_idx = kept_indices[self.state_noise_idx]
        if len(new_indices) > 0:
            next_state_vec[new_indices] = new_state_vec
            next_state_cov[new_indices] = new_state_cov
            next_noise_var[new_indices] = new_noise_var
        self.state_vec = next_state_vec
        self.state_cov = next_state_cov
        self.noise_var = next_noise_var

    def deep_copy(self):
        '''Return a deep copy of the state'''
        c = KalmanState(self.observation_matrix, self.translation_matrix)
        c.state_vec = self.state_vec.copy()
        c.state_cov = self.state_cov.copy()
        c.noise_var = self.noise_var.copy()
        c.state_noise = self.state_noise.copy()
        c.state_noise_idx = self.state_noise_idx.copy()
        return c

LARGE_KALMAN_COV = 2000
SMALL_KALMAN_COV = 1

def velocity_kalman_model():
    '''Return a KalmanState set up to model objects with constant velocity

    The observation and measurement vectors are i,j.
    The state vector is i,j,vi,vj
    '''
    om = np.array([[1,0,0,0], [0, 1, 0, 0]])
    tm = np.array([[1,0,1,0],
                   [0,1,0,1],
                   [0,0,1,0],
                   [0,0,0,1]])
    return KalmanState(om, tm)

def reverse_velocity_kalman_model():
    '''Return a KalmanState set up to model going backwards in time'''
    om = np.array([[1,0,0,0], [0, 1, 0, 0]])
    tm = np.array([[1,0,-1,0],
                   [0,1,0,-1],
                   [0,0,1,0],
                   [0,0,0,1]])
    return KalmanState(om, tm)

def static_kalman_model():
    '''Return a KalmanState set up to model objects whose motion is random

    The observation, measurement and state vectors are all i,j
    '''
    return KalmanState(np.eye(2), np.eye(2))

def kalman_filter(kalman_state, old_indices, coordinates, q, r):
    '''Return the kalman filter for the features in the new frame

    kalman_state - state from last frame

    old_indices - the index per feature in the last frame or -1 for new

    coordinates - Coordinates of the features in the new frame.

    q - the process error covariance - see equ 1.3 and 1.10 from Welch

    r - measurement error covariance of features - see eqn 1.7 and 1.8 from welch.

    returns a new KalmanState containing the kalman filter of
    the last state by the given coordinates.

    Refer to kalmanGainLinearMotion.m and
    http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

    for info on the algorithm.
    '''
    assert isinstance(kalman_state, KalmanState)
    old_indices = np.array(old_indices)
    if len(old_indices) == 0:
        return KalmanState(kalman_state.observation_matrix,
                           kalman_state.translation_matrix)
    #
    # Cull missing features in old state and collect only matching coords
    #
    matching = old_indices != -1
    new_indices = np.arange(len(old_indices))[~matching]
    retained_indices = np.arange(len(old_indices))[matching]
    new_coords = coordinates[new_indices]
    observation_matrix_t = kalman_state.observation_matrix.transpose()
    if len(retained_indices) > 0:
        kalman_state = kalman_state.deep_copy()
        coordinates = coordinates[retained_indices]
        kalman_state.map_frames(old_indices[retained_indices])
        #
        # Time update equations
        #
        # From eqn 1.9 of Welch
        #
        state_vec = kalman_state.predicted_state_vec
        #
        # From eqn 1.10 of Welch
        #
        state_cov = dot_n(
            dot_n(kalman_state.translation_matrix, kalman_state.state_cov),
            kalman_state.translation_matrix.transpose()) + q[matching]
        #
        # From eqn 1.11 of welch
        #
        kalman_gain_numerator = dot_n(state_cov, observation_matrix_t)

        kalman_gain_denominator = dot_n(
            dot_n(kalman_state.observation_matrix, state_cov),
                  observation_matrix_t) + r[matching]
        kalman_gain_denominator = inv_n(kalman_gain_denominator)
        kalman_gain = dot_n(kalman_gain_numerator, kalman_gain_denominator)
        #
        # Eqn 1.12 of Welch
        #
        difference = coordinates - dot_n(kalman_state.observation_matrix,
                                         state_vec[:,:,np.newaxis])[:,:,0]
        state_noise = dot_n(kalman_gain, difference[:,:,np.newaxis])[:,:,0]
        state_vec = state_vec + state_noise
        #
        # Eqn 1.13 of Welch (factored from (I - KH)P to P - KHP)
        #
        state_cov = (state_cov -
                     dot_n(dot_n(kalman_gain, kalman_state.observation_matrix),
                           state_cov))
        #
        # Collect all of the state noise in one array. We produce an I and J
        # variance. Notes in kalmanGainLinearMotion indicate that you
        # might want a single variance, combining I & J. An alternate
        # might be R and theta, a variance of angular consistency and one
        # of absolute velocity.
        #
        # Add an index to the state noise in the rightmost column
        #
        idx = np.arange(len(state_noise))
        #
        # Stack the rows with the old ones
        #
        all_state_noise = np.vstack((kalman_state.state_noise, state_noise))
        all_state_noise_idx = np.hstack((kalman_state.state_noise_idx, idx))
        noise_var = np.zeros((len(idx), all_state_noise.shape[1]))
        for i in range(all_state_noise.shape[1]):
            noise_var[:, i] = fix(scind.variance(all_state_noise[:, i],
                                                 all_state_noise_idx,
                                                 idx))
        obs_vec = dot_n(kalman_state.observation_matrix,
                        state_vec[:,:,np.newaxis])[:,:,0]
        kalman_state = KalmanState(kalman_state.observation_matrix,
                                   kalman_state.translation_matrix,
                                   state_vec, state_cov, noise_var,
                                   all_state_noise,
                                   all_state_noise_idx)
    else:
        # Erase all previous features
        kalman_state = KalmanState(kalman_state.observation_matrix,
                                   kalman_state.translation_matrix)
    if len(new_coords) > 0:
        #
        # Fill in the initial states:
        #
        state_vec = dot_n(observation_matrix_t,
                          new_coords[:,:,np.newaxis])[:,:,0]
        #
        # The COV for the hidden, undetermined features should be large
        # and the COV for others should be small
        #
        nstates = kalman_state.state_len
        nnew_features = len(new_indices)
        cov_vec = SMALL_KALMAN_COV / np.dot(observation_matrix_t,
                                            np.ones(kalman_state.obs_len))
        cov_vec[~ np.isfinite(cov_vec)] = LARGE_KALMAN_COV
        cov_matrix = np.diag(cov_vec)
        state_cov = cov_matrix[np.newaxis,:,:][np.zeros(nnew_features,int)]
        #
        # The noise variance is all ones in Jaqman
        #
        noise_var = np.ones((len(new_indices), kalman_state.state_len))
        #
        # Map the retained indices to their new slots and new ones to empty
        # slots (=-1)
        #
        kalman_state.add_features(retained_indices,
                                  new_indices,
                                  state_vec, state_cov, noise_var)
    return kalman_state

def line_integration(image, angle, decay, sigma):
    '''Integrate the image along the given angle

    DIC images are the directional derivative of the underlying
    image. This filter reconstructs the original image by integrating
    along that direction.

    image - a 2-dimensional array

    angle - shear angle in radians. We integrate perpendicular to this angle

    decay - an exponential decay applied to the integration

    sigma - the standard deviation of a Gaussian which is used to
            smooth the image in the direction parallel to the shear angle.
    '''
    #
    # Normalize the image so that the mean is zero
    #
    normalized = image - np.mean(image)
    #
    # Rotate the image so the J direction is perpendicular to the shear angle.
    #
    rotated = scind.rotate(normalized, -angle)
    #
    # Smooth in only the i direction
    #
    smoothed = scind.gaussian_filter1d(rotated, sigma) if sigma > 0 else rotated
    #
    # We want img_out[:,j+1] to be img_out[:,j] * decay + img[j+1]
    # Could be done by convolution with a ramp, maybe in FFT domain,
    # but we just do a bunch of steps here.
    #
    result_fwd = smoothed.copy()
    for i in range(1,result_fwd.shape[0]):
        result_fwd[i] += result_fwd[i-1] * decay
    result_rev = smoothed.copy()
    for i in reversed(range(result_rev.shape[0]-1)):
        result_rev[i] += result_rev[i+1] * decay
    result = (result_fwd - result_rev) / 2
    #
    # Rotate and chop result
    #
    result = scind.rotate(result, angle)
    ipad = int((result.shape[0] - image.shape[0]) / 2)
    jpad = int((result.shape[1] - image.shape[1]) / 2)
    result = result[ipad:(ipad + image.shape[0]),
                    jpad:(jpad + image.shape[1])]
    #
    # Scale the resultant image similarly to the output.
    #
    img_min, img_max = np.min(image), np.max(image)
    result_min, result_max = np.min(result), np.max(result)
    if (img_min == img_max) or (result_min == result_max):
        return np.zeros(result.shape)
    result = (result - result_min) / (result_max - result_min)
    result = img_min + result * (img_max - img_min)
    return result

def variance_transform(img, sigma, mask=None):
    '''Calculate a weighted variance of the image

    This function caluclates the variance of an image, weighting the
    local contributions by a Gaussian.

    img - image to be transformed
    sigma - standard deviation of the Gaussian
    mask - mask of relevant pixels in the image
    '''
    if mask is None:
        mask = np.ones(img.shape, bool)
    else:
        img = img.copy()
        img[~mask] = 0
    #
    # This is the Gaussian of the mask... so we can normalize for
    # pixels near the edge of the mask
    #
    gmask = scind.gaussian_filter(mask.astype(float), sigma,
                                  mode = 'constant')
    img_mean = scind.gaussian_filter(img, sigma,
                                     mode = 'constant') / gmask
    img_squared = scind.gaussian_filter(img ** 2, sigma,
                                        mode = 'constant') / gmask
    var = img_squared - img_mean ** 2
    return var
    #var = var[kernel_half_width:(kernel_half_width + img.shape[0]),
              #kernel_half_width:(kernel_half_width + img.shape[0])]
    #ik = ik.ravel()
    #jk = jk.ravel()
    #gk = np.exp(-(ik*ik + jk*jk) / (2 * sigma * sigma))
    #gk = (gk / np.sum(gk)).astype(np.float32)
    ## We loop here in chunks of 32 x 32 because the kernel can get large.
    ## Remove this loop in 2025 when Numpy can grok the big object itself
    ## and construct the loop and run it on 1,000,000 GPU cores
    ##
    #var = np.zeros(img.shape, np.float32)
    #for ioff in range(0, img.shape[0], 32):
        #for joff in range(0, img.shape[1], 32):
            ##
            ## ib and jb give addresses of the center pixel in the big image
            ##
            #iend = min(ioff+32, img.shape[0])
            #jend = min(joff+32, img.shape[1])
            #ii = np.arange(ioff, iend)
            #ib = ii + kernel_half_width
            #jj = np.arange(joff, jend)
            #jb = jj + kernel_half_width
            ##
            ## Axes 0 and 1 are the axes of the final array and rely on ib and jb
            ## to find the centers of the kernels in the big image.
            ##
            ## Axis 2 iterates over the elements and offsets in the kernel.
            ##
            ## We multiply each kernel contribution by the Gaussian gk to weight
            ## the kernel pixel's contribution. We multiply each contribution
            ## by its truth value in the mask to cross out border pixels.
            ##
            #norm_chunk = (
                #big_img[
                    #ib[:,np.newaxis,np.newaxis] + ik[np.newaxis, np.newaxis,:],
                    #jb[np.newaxis,:,np.newaxis] + jk[np.newaxis, np.newaxis,:]] -
                #img_mean[ib[:,np.newaxis,np.newaxis],
                         #jb[np.newaxis,:,np.newaxis]])

            #var[ii[:,np.newaxis],jj[np.newaxis,:]] = np.sum(
                #norm_chunk * norm_chunk *
                #gk[np.newaxis, np.newaxis,:] *
                #big_mask[ib[:,np.newaxis,np.newaxis] +
                         #ik[np.newaxis, np.newaxis,:],
                         #jb[np.newaxis,:,np.newaxis] +
                         #jk[np.newaxis, np.newaxis,:]], 2)
    ##
    ## Finally, we divide by the Gaussian of the mask to normalize for
    ## pixels without contributions from masked pixels in their kernel.
    ##
    #var /= gmask[kernel_half_width:(kernel_half_width+var.shape[0]),
                 #kernel_half_width:(kernel_half_width+var.shape[1])]
    #return var

def inv_n(x):
    '''given N matrices, return N inverses'''
    #
    # The inverse of a small matrix (e.g. 3x3) is
    #
    #   1
    # -----   C(j,i)
    # det(A)
    #
    # where C(j,i) is the cofactor of matrix A at position j,i
    #
    assert x.ndim == 3
    assert x.shape[1] == x.shape[2]
    c = np.array([ [cofactor_n(x, j, i) * (1 - ((i+j) % 2)*2)
                    for j in range(x.shape[1])]
                   for i in range(x.shape[1])]).transpose(2,0,1)
    return c / det_n(x)[:, np.newaxis, np.newaxis]

def det_n(x):
    '''given N matrices, return N determinants'''
    assert x.ndim == 3
    assert x.shape[1] == x.shape[2]
    if x.shape[1] == 1:
        return x[:,0,0]
    result = np.zeros(x.shape[0])
    for permutation in permutations(np.arange(x.shape[1])):
        sign = parity(permutation)
        result += np.prod([x[:, i, permutation[i]]
                           for i in range(x.shape[1])], 0) * sign
        sign = - sign
    return result

def parity(x):
    '''The parity of a permutation

    The parity of a permutation is even if the permutation can be
    formed by an even number of transpositions and is odd otherwise.

    The parity of a permutation is even if there are an even number of
    compositions of even size and odd otherwise. A composition is a cycle:
    for instance in (1, 2, 0, 3), there is the cycle: (0->1, 1->2, 2->0)
    and the cycle, (3->3). Both cycles are odd, so the parity is even:
    you can exchange 0 and 1 giving (0, 2, 1, 3) and 2 and 1 to get
    (0, 1, 2, 3)
    '''

    order = np.lexsort((x,))
    hit = np.zeros(len(x), bool)
    p = 0
    for j in range(len(x)):
        if not hit[j]:
            cycle = 1
            i = order[j]
            # mark every node in a cycle
            while i != j:
                hit[i] = True
                i = order[i]
                cycle += 1
            p += cycle - 1
    return 1 if p % 2 == 0 else -1

def cofactor_n(x, i, j):
    '''Return the cofactor of n matrices x[n,i,j] at position i,j

    The cofactor is the determinant of the matrix formed by removing
    row i and column j.
    '''
    m = x.shape[1]
    mr = np.arange(m)
    i_idx = mr[mr != i]
    j_idx = mr[mr != j]
    return det_n(x[:, i_idx[:, np.newaxis],
                      j_idx[np.newaxis, :]])

def dot_n(x, y):
    '''given two tensors N x I x K and N x K x J return N dot products

    If either x or y is 2-dimensional, broadcast it over all N.

    Dot products are size N x I x J.
    Example:
    x = np.array([[[1,2], [3,4], [5,6]],[[7,8], [9,10],[11,12]]])
    y = np.array([[[1,2,3], [4,5,6]],[[7,8,9],[10,11,12]]])
    print dot_n(x,y)

    array([[[  9,  12,  15],
        [ 19,  26,  33],
        [ 29,  40,  51]],

       [[129, 144, 159],
        [163, 182, 201],
        [197, 220, 243]]])
    '''
    if x.ndim == 2:
        if y.ndim == 2:
            return np.dot(x, y)
        x3 = False
        y3 = True
        nlen = y.shape[0]
    elif y.ndim == 2:
        nlen = x.shape[0]
        x3 = True
        y3 = False
    else:
        assert x.shape[0] == y.shape[0]
        nlen = x.shape[0]
        x3 = True
        y3 = True
    assert x.shape[1+x3] == y.shape[0+y3]
    n, i, j, k = np.mgrid[0:nlen, 0:x.shape[0+x3], 0:y.shape[1+y3],
                          0:y.shape[0+y3]]
    return np.sum((x[n, i, k] if x3 else x[i,k]) *
                  (y[n, k, j] if y3 else y[k,j]), 3)

def permutations(x):
    '''Given a listlike, x, return all permutations of x

    Returns the permutations of x in the lexical order of their indices:
    e.g.
    >>> x = [ 1, 2, 3, 4 ]
    >>> for p in permutations(x):
    >>>   print p
    [ 1, 2, 3, 4 ]
    [ 1, 2, 4, 3 ]
    [ 1, 3, 2, 4 ]
    [ 1, 3, 4, 2 ]
    [ 1, 4, 2, 3 ]
    [ 1, 4, 3, 2 ]
    [ 2, 1, 3, 4 ]
    ...
    [ 4, 3, 2, 1 ]
    '''
    #
    # The algorithm is attributed to Narayana Pandit from his
    # Ganita Kaumundi (1356). The following is from
    #
    # http://en.wikipedia.org/wiki/Permutation#Systematic_generation_of_all_permutations
    #
    # 1. Find the largest index k such that a[k] < a[k + 1].
    #    If no such index exists, the permutation is the last permutation.
    # 2. Find the largest index l such that a[k] < a[l].
    #    Since k + 1 is such an index, l is well defined and satisfies k < l.
    # 3. Swap a[k] with a[l].
    # 4. Reverse the sequence from a[k + 1] up to and including the final
    #    element a[n].
    #
    yield list(x) # don't forget to do the first one
    x = np.array(x)
    a = np.arange(len(x))
    while True:
        # 1 - find largest or stop
        ak_lt_ak_next = np.argwhere(a[:-1] < a[1:])
        if len(ak_lt_ak_next) == 0:
            raise StopIteration()
        k = ak_lt_ak_next[-1, 0]
        # 2 - find largest a[l] < a[k]
        ak_lt_al = np.argwhere(a[k] < a)
        l =  ak_lt_al[-1, 0]
        # 3 - swap
        a[k], a[l]  = (a[l], a[k])
        # 4 - reverse
        if k < len(x)-1:
            a[k+1:] = a[:k:-1].copy()
        yield x[a].tolist()

def convex_hull_transform(image, levels=256, mask = None,
                          chunksize = CONVEX_HULL_CHUNKSIZE,
                          pass_cutoff = 16):
    '''Perform the convex hull transform of this image

    image - image composed of integer intensity values
    levels - # of levels that we separate the image into
    mask - mask of points to consider or None to consider all points
    chunksize - # of points processed in first pass of convex hull

    for each intensity value, find the convex hull of pixels at or above
    that value and color all pixels within the hull with that value.
    '''
    # Scale the image into the requisite number of levels
    if mask is None:
        img_min = np.min(image)
        img_max = np.max(image)
    else:
        unmasked_pixels = image[mask]
        if len(unmasked_pixels) == 0:
            return np.zeros(image.shape, image.dtype)
        img_min = np.min(unmasked_pixels)
        img_max = np.max(unmasked_pixels)
    img_shape = tuple(image.shape)
    if img_min == img_max:
        return image

    scale = (img_min +
             np.arange(levels).astype(image.dtype) *
             (img_max - img_min) / float(levels-1))
    image = ((image - img_min) * (levels-1) / (img_max - img_min))
    if mask is not None:
        image[~mask] = 0
    #
    # If there are more than 16 levels, we do the method first at a coarse
    # scale. The dark objects can produce points at every level, so doing
    # two passes can reduce the number of points in the second pass to
    # only the difference between two levels at the coarse pass.
    #
    if levels > pass_cutoff:
        sub_levels = int(np.sqrt(levels))
        rough_image = convex_hull_transform(np.floor(image), sub_levels)
        image = np.maximum(image, rough_image)
        del rough_image
    image = image.astype(int)
    #
    # Get rid of any levels that have no representatives
    #
    unique = np.unique(image)
    new_values = np.zeros(levels, int)
    new_values[unique] = np.arange(len(unique))
    scale = scale[unique]
    image = new_values[image]
    #
    # Start by constructing the list of points which are local maxima
    #
    min_image = grey_erosion(image, footprint = np.ones((3,3), bool)).astype(int)
    #
    # Set the borders of the min_image to zero so that the border pixels
    # will be in all convex hulls below their intensity
    #
    min_image[0, :] = 0
    min_image[-1, :] = 0
    min_image[:, 0] = 0
    min_image[:, -1] = 0

    i,j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    mask = image > min_image

    i = i[mask]
    j = j[mask]
    min_image = min_image[mask]
    image = image[mask]
    #
    # Each point that is a maximum is a potential vertex in the convex hull
    # for each value above the minimum. Therefore, it appears
    #
    # image - min_image
    #
    # times in the i,j,v list of points. So we can do a sum to calculate
    # the number of points, then use cumsum to figure out the first index
    # in that array of points for each i,j,v. We can then use cumsum
    # again on the array of points to assign their levels.
    #
    count = image - min_image
    npoints = np.sum(count)
    # The index in the big array of the first point to place for each
    # point
    first_index_in_big = np.cumsum(count) - count
    #
    # The big array can be quite big, for example if there are lots of
    # thin, dark objects. We do two passes of convex hull: the convex hull
    # of the convex hulls of several regions is the convex hull of the whole
    # so it doesn't matter too much how we break up the array.
    #
    first_i = np.zeros(0, int)
    first_j = np.zeros(0, int)
    first_levels = np.zeros(0, int)
    chunkstart = 0
    while chunkstart < len(count):
        idx = first_index_in_big[chunkstart]
        iend = idx + chunksize
        if iend >= npoints:
            chunkend = len(count)
            iend = npoints
        else:
            chunkend = np.searchsorted(first_index_in_big, iend)
            if chunkend < len(count):
                iend = first_index_in_big[chunkend]
            else:
                iend = npoints
        chunk_first_index_in_big = first_index_in_big[chunkstart:chunkend] - idx
        chunkpoints = iend - idx
        #
        # For the big array, construct an array of indexes into the small array
        #
        index_in_small = np.zeros(chunkpoints, int)
        index_in_small[0] = chunkstart
        index_in_small[chunk_first_index_in_big[1:]] = 1
        index_in_small = np.cumsum(index_in_small)
        #
        # We're going to do a cumsum to make the big array of levels. Point
        # n+1 broadcasts its first value into first_index_in_big[n+1].
        # The value that precedes it is image[n]. Therefore, in order to
        # get the correct value in cumsum:
        #
        # ? + image[n] = min_image[n+1]+1
        # ? = min_image[n+1] + 1 - image[n]
        #
        levels = np.ones(chunkpoints, int)
        levels[0] = min_image[chunkstart] + 1
        levels[chunk_first_index_in_big[1:]] = \
            min_image[chunkstart+1:chunkend] - image[chunkstart:chunkend-1] + 1
        levels = np.cumsum(levels)
        #
        # Construct the ijv
        #
        ijv = np.column_stack((i[index_in_small], j[index_in_small], levels))
        #
        # Get all of the convex hulls
        #
        pts, counts = convex_hull_ijv(ijv, np.arange(1, len(unique)))
        first_i = np.hstack((first_i, pts[:, 1]))
        first_j = np.hstack((first_j, pts[:, 2]))
        first_levels = np.hstack((first_levels, pts[:, 0]))
        chunkstart = chunkend
    #
    # Now do the convex hull of the reduced list of points
    #
    ijv = np.column_stack((first_i, first_j, first_levels))
    pts, counts = convex_hull_ijv(ijv, np.arange(1, len(unique)))
    #
    # Get the points along the lines described by the convex hulls
    #
    # There are N points for each label. Draw a line from each to
    # the next, except for the last which we draw from last to first
    #
    labels = pts[:, 0]
    i = pts[:, 1]
    j = pts[:, 2]
    first_index = np.cumsum(counts) - counts
    last_index = first_index + counts - 1
    next = np.arange(len(labels)) + 1
    next[last_index] = first_index
    index, count, i, j = get_line_pts(i, j, i[next], j[next])
    #
    # use a cumsum to get the index of each point from get_line_pts
    # relative to the labels vector
    #
    big_index = np.zeros(len(i), int)
    big_index[index[1:]] = 1
    big_index = np.cumsum(big_index)
    labels = labels[big_index]
    #
    # A given i,j might be represented more than once. Take the maximum
    # label at each i,j. First sort by i,j and label. Then take only values
    # that have a different i,j than the succeeding value. The last element
    # is always a winner.
    #
    order = np.lexsort((labels, i, j))
    i = i[order]
    j = j[order]
    labels = labels[order]
    mask = np.hstack((((i[1:] != i[:-1]) | (j[1:] != j[:-1])),[True]))
    i = i[mask]
    j = j[mask]
    labels = labels[mask]
    #
    # Now, we have an interesting object. It's ordered by j, then i which
    # means that we have scans of interesting i at each j. The points
    # that aren't represented should have the minimum of the values
    # above and below.
    #
    # We can play a cumsum trick to do this, placing the difference
    # of a point with its previous in the 2-d image, then summing along
    # each i axis to set empty values to the value of the nearest occupied
    # value above and a similar trick to set empty values to the nearest
    # value below. We then take the minimum of the two results.
    #
    first = np.hstack(([True], j[1:] != j[:-1]))
    top = np.zeros(img_shape, labels.dtype)
    top[i[first], j[first]] = labels[first]
    top[i[~first], j[~first]] = (labels[1:] - labels[:-1])[~first[1:]]
    top = np.cumsum(top, 0)

    # From 0 to the location of the first point, set to value of first point
    bottom = np.zeros(img_shape, labels.dtype)
    bottom[0, j[first]] = labels[first]
    # From 1 + the location of the previous point, set to the next point
    last = np.hstack((first[1:], [True]))
    bottom[i[:-1][~first[1:]]+1, j[~first]] = (labels[1:] - labels[:-1])[~first[1:]]
    # Set 1 + the location of the last point to -labels so that all past
    # the end will be zero (check for i at end...)
    llast = last & (i < img_shape[0] - 1)
    bottom[i[llast]+1, j[llast]] = - labels[llast]
    bottom = np.cumsum(bottom, 0)
    image = np.minimum(top, bottom)
    return scale[image]

def circular_hough(img, radius, nangles = None, mask=None):
    '''Circular Hough transform of an image

    img - image to be transformed.

    radius - radius of circle

    nangles - # of angles to measure, e.g. nangles = 4 means accumulate at
              0, 90, 180 and 270 degrees.

    Return the Hough transform of the image which is the accumulators
    for the transform x + r cos t, y + r sin t.
    '''
    a = np.zeros(img.shape)
    m = np.zeros(img.shape)
    if nangles is None:
        # if no angle specified, take the circumference
        # Round to a multiple of 4 to make it bilaterally stable
        nangles = int(np.pi * radius + 3.5) & (~ 3)
    for i in range(nangles):
        theta = 2*np.pi * float(i) / float(nangles)
        x = int(np.round(radius * np.cos(theta)))
        y = int(np.round(radius * np.sin(theta)))
        xmin = max(0, -x)
        xmax = min(img.shape[1] - x, img.shape[1])
        ymin = max(0, -y)
        ymax = min(img.shape[0] - y, img.shape[0])
        dest = (slice(ymin, ymax),
                 slice(xmin, xmax))
        src = (slice(ymin+y, ymax+y), slice(xmin+x, xmax+x))
        if mask is not None:
            a[dest][mask[src]] += img[src][mask[src]]
            m[dest][mask[src]] += 1
        else:
            a[dest] += img[src]
            m[dest] += 1
    a[m > 0] /= m[m > 0]
    return a

def hessian(image, return_hessian=True, return_eigenvalues=True, return_eigenvectors=True):
    '''Calculate hessian, its eigenvalues and eigenvectors

    image - n x m image. Smooth the image with a Gaussian to get derivatives
            at different scales.

    return_hessian - true to return an n x m x 2 x 2 matrix of the hessian
                     at each pixel

    return_eigenvalues - true to return an n x m x 2 matrix of the eigenvalues
                         of the hessian at each pixel

    return_eigenvectors - true to return an n x m x 2 x 2 matrix of the
                          eigenvectors of the hessian at each pixel

    The values of the border pixels for the image are not calculated and
    are zero
    '''
    #The Hessian, d(f(x0, x1))/dxi/dxj for i,j = [0,1] is approximated by the
    #following kernels:

    #d00: [[1], [-2], [1]]
    #d11: [[1, -2, 1]]
    #d01 and d10: [[   1, 0,-1],
                  #[   0, 0, 0],
                  #[  -1, 0, 1]] / 2


    #The eigenvalues of the hessian:
    #[[d00, d01]
     #[d01, d11]]
     #L1 = (d00 + d11) / 2 + ((d00 + d11)**2 / 4 - (d00 * d11 - d01**2)) ** .5
     #L2 = (d00 + d11) / 2 - ((d00 + d11)**2 / 4 - (d00 * d11 - d01**2)) ** .5

     #The eigenvectors of the hessian:
     #if d01 != 0:
       #[(L1 - d11, d01), (L2 - d11, d01)]
    #else:
       #[ (1, 0), (0, 1) ]


    #Ideas and code borrowed from:
    #http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
    #http://www.longair.net/edinburgh/imagej/tubeness/


    hessian = np.zeros((image.shape[0], image.shape[1], 2, 2))
    hessian[1:-1, :, 0, 0] = image[:-2, :] - (2 * image[1:-1, :]) + image[2:, :]
    hessian[1:-1, 1:-1, 0, 1] = hessian[1:-1, 1:-1, 0, 1] = (
        image[2:, 2:] + image[:-2, :-2] -
        image[2:, :-2] - image[:-2, 2:]) / 4
    hessian[:, 1:-1, 1, 1] = image[:, :-2] - (2 * image[:, 1:-1]) + image[:, 2:]
    #
    # Solve the eigenvalue equation:
    # H x = L x
    #
    # Much of this from Eigensystem2x2Float.java from tubeness
    #
    A = hessian[:, :, 0, 0]
    B = hessian[:, :, 0, 1]
    C = hessian[:, :, 1, 1]

    b = -(A + C)
    c = A * C - B * B
    discriminant = b * b - 4 * c

    # pn is something that broadcasts over all points and either adds or
    # subtracts the +/- part of the eigenvalues

    pn = np.array([1, -1])[np.newaxis, np.newaxis, :]
    L = (- b[:, :, np.newaxis] +
         (np.sqrt(discriminant)[:, :, np.newaxis] * pn)) / 2
    #
    # Report eigenvalue # 0 as the one with the highest absolute magnitude
    #
    L[np.abs(L[:, :, 1]) > np.abs(L[:, :, 0]), :] =\
      L[np.abs(L[:, :, 1]) > np.abs(L[:, :, 0]), ::-1]


    if return_eigenvectors:
        #
        # Calculate for d01 != 0
        #
        v = np.ones((image.shape[0], image.shape[1], 2, 2)) * np.nan
        v[:, :, :, 0] =  L - hessian[:, :, 1, 1, np.newaxis]
        v[:, :, :, 1] = hessian[:, :, 0, 1, np.newaxis]
        #
        # Calculate for d01 = 0
        default = np.array([[1, 0], [0, 1]])[np.newaxis, :, :]
        v[hessian[:, :, 0, 1] == 0] = default
        #
        # Normalize the vectors
        #
        d = np.sqrt(np.sum(v * v, 3))
        v /= d[:, :, :, np.newaxis]

    result = []
    if return_hessian:
        result.append(hessian)
    if return_eigenvalues:
        result.append(L)
    if return_eigenvectors:
        result.append(v)
    if len(result) == 0:
        return
    elif len(result) == 1:
        return result[0]
    return tuple(result)

def poisson_equation(image, gradient=1, max_iter=100, convergence=.01, percentile = 90.0):
    '''Estimate the solution to the Poisson Equation

    The Poisson Equation is the solution to gradient(x) = h^2/4 and, in this
    context, we use a boundary condition where x is zero for background
    pixels. Also, we set h^2/4 = 1 to indicate that each pixel is a distance
    of 1 from its neighbors.

    The estimation exits after max_iter iterations or if the given percentile
    of foreground pixels differ by less than the convergence fraction
    from one pass to the next.

    Some ideas taken from Gorelick, "Shape representation and classification
    using the Poisson Equation", IEEE Transactions on Pattern Analysis and
    Machine Intelligence V28, # 12, 2006

    image - binary image with foreground as True
    gradient - the target gradient between 4-adjacent pixels
    max_iter - maximum # of iterations at a given level
    convergence - target fractional difference between values from previous
                  and next pass
    percentile - measure convergence at this percentile
    '''
    # Evaluate the poisson equation with zero-padded boundaries
    pe = np.zeros((image.shape[0]+2, image.shape[1]+2))
    if image.shape[0] > 64 and image.shape[1] > 64:
        #
        # Sub-sample to get seed values
        #
        sub_image = image[::2, ::2]
        sub_pe = poisson_equation(sub_image,
                                  gradient=gradient*2,
                                  max_iter=max_iter,
                                  convergence=convergence)
        coordinates = np.mgrid[0:(sub_pe.shape[0]*2),
                               0:(sub_pe.shape[1]*2)].astype(float) / 2
        pe[1:(sub_image.shape[0]*2+1), 1:(sub_image.shape[1]*2+1)] = \
            scind.map_coordinates(sub_pe, coordinates, order=1)
        pe[:image.shape[0], :image.shape[1]][~image] = 0
    else:
        pe[1:-1,1:-1] = image
    #
    # evaluate only at i and j within the foreground
    #
    i, j = np.mgrid[0:pe.shape[0], 0:pe.shape[1]]
    mask = (i>0) & (i<pe.shape[0]-1) & (j>0) & (j<pe.shape[1]-1)
    mask[mask] = image[i[mask]-1, j[mask]-1]
    i = i[mask]
    j = j[mask]
    if len(i) == 0:
        return pe[1:-1, 1:-1]
    if len(i) == 1:
        # Just in case "percentile" can't work when unable to interpolate
        # between a single value... Isolated pixels have value = 1
        #
        pe[mask] = 1
        return pe[1:-1, 1:-1]

    for itr in range(max_iter):
        next_pe = (pe[i+1, j] + pe[i-1, j] + pe[i, j+1] + pe[i, j-1]) / 4 + 1
        difference = np.abs((pe[mask] - next_pe) / next_pe)
        pe[mask] = next_pe
        if np.percentile(difference, percentile) <= convergence:
            break
    return pe[1:-1, 1:-1]

