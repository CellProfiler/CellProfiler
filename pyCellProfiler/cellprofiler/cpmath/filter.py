'''filter.py - functions for applying filters to images

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision: 1 "

import numpy as np
import _filter
from rankorder import rank_order
from scipy.ndimage import map_coordinates
from scipy.ndimage import convolve

def stretch(image, mask=None):
    '''Normalize an image to make the minimum zero and maximum one
    
    image - pixel data to be normalized
    mask  - optional mask of relevant pixels. None = don't mask
    
    returns the stretched image
    '''
    if np.product(image.shape) == 0:
        return image
    if mask is None:
        minval = np.min(image)
        maxval = np.max(image)
        if minval == maxval:
            return image
        else:
            return (image - minval) / (maxval - minval)
    else:
        significant_pixels = image[mask]
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

def median_filter(data, mask, radius, percent=50):
    '''Masked median filter with octagonal shape
    
    data - array of data to be median filtered.
    mask - mask of significant pixels in data
    radius - the radius of a circle inscribed into the filtering octagon
    percent - conceptually, order the significant pixels in the octagon,
              count them and choose the pixel indexed by the percent
              times the count divided by 100. More simply, 50 = median
    returns a filtered array
    '''
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
    not_mask = np.logical_not(mask)
    result[not_mask] = data[not_mask] 
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