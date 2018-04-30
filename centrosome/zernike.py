from __future__ import absolute_import
from __future__ import division
import numpy as np
import scipy.sparse
import scipy.ndimage

from .cpmorphology import minimum_enclosing_circle,fixup_scipy_ndimage_result
from .cpmorphology import fill_labeled_holes,draw_line
from six.moves import range
from six.moves import zip


def construct_zernike_lookuptable(zernike_indexes):
    """Return a lookup table of the sum-of-factorial part of the radial
    polynomial of the zernike indexes passed
    
    zernike_indexes - an Nx2 array of the Zernike polynomials to be
                      computed.
    """
    n_max = np.max(zernike_indexes[:,0])
    factorial = np.ones((1 + n_max,), dtype=float)
    factorial[1:] = np.cumproduct(np.arange(1, 1 + n_max, dtype=float))
    width = int(n_max//2 + 1)
    lut = np.zeros((zernike_indexes.shape[0],width), dtype=float)
    for idx, (n, m) in enumerate(zernike_indexes):
        alt = 1
        npmh = (n+m)//2
        nmmh = (n-m)//2
        for k in range(0,nmmh+1):
            lut[idx,k] = \
                (alt * factorial[n-k] /
                 (factorial[k]*factorial[npmh-k]*factorial[nmmh-k]))
            alt = -alt
    return lut


def construct_zernike_polynomials(x, y, zernike_indexes, mask=None, weight=None):
    """Return the zerike polynomials for all objects in an image
    
    x - the X distance of a point from the center of its object
    y - the Y distance of a point from the center of its object
    zernike_indexes - an Nx2 array of the Zernike polynomials to be computed.
    mask - a mask with same shape as X and Y of the points to consider
    weight - weightings of points with the same shape as X and Y (default
             weight on each point is 1).
    
    returns a height x width x N array of complex numbers which are the
    e^i portion of the sine and cosine of the Zernikes
    """
    if x.shape != y.shape:
        raise ValueError("X and Y must have the same shape")
    if mask is None:
        pass
    elif mask.shape != x.shape:
        raise ValueError("The mask must have the same shape as X and Y")
    else:
        x = x[mask]
        y = y[mask]
        if weight is not None:
            weight = weight[mask]
    lut = construct_zernike_lookuptable(zernike_indexes) # precompute poly. coeffs.
    nzernikes = zernike_indexes.shape[0]
    # compute radii
    r_square = np.square(x) # r_square = x**2
    np.add(r_square, np.square(y), out=r_square) # r_square = x**2 + y**2
    # z = y + 1j*x
    # each Zernike polynomial is poly(r)*(r**m * np.exp(1j*m*phi)) ==
    #                            poly(r)*(y + 1j*x)**m
    z = np.empty(x.shape, np.complex)
    np.copyto(z.real, y)
    np.copyto(z.imag, x)
    # preallocate buffers
    s = np.empty_like(x)
    zf = np.zeros((nzernikes,) + x.shape, np.complex)
    z_pows = {}
    for idx, (n, m) in enumerate(zernike_indexes):
        s[:]=0
        if not m in z_pows:
            if m == 0:
                z_pows[m] = np.complex(1.0)
            else:
                z_pows[m] = z if m == 1 else (z ** m)
        z_pow = z_pows[m]
        # use Horner scheme
        for k in range((n-m)//2+1):
            s *= r_square
            s += lut[idx, k]
        s[r_square>1]=0
        if weight is not None:
            s *= weight.astype(s.dtype)
        if m == 0:
            np.copyto(zf[idx], s) # zf[idx] = s
        else:
            np.multiply(s, z_pow, out=zf[idx]) # zf[idx] = s*exp_term
    
    if mask is None:
        result = zf.transpose( tuple(range(1, 1+x.ndim)) + (0, ))
    else:
        result = np.zeros( mask.shape + (nzernikes,), np.complex)
        result[mask] = zf.transpose( tuple(range(1, 1 + x.ndim)) + (0, ))
    return result


def score_zernike(zf, radii, labels, indexes=None):
    """Score the output of construct_zernike_polynomials
    
    zf - the output of construct_zernike_polynomials which is I x J x K
         where K is the number of zernike polynomials computed
    radii - a vector of the radius of each of N labeled objects
    labels - a label matrix
    
    outputs a N x K matrix of the scores of each of the Zernikes for
    each labeled object.
    """
    if indexes is None:
        indexes = np.arange(1,np.max(labels)+1,dtype=np.int32)
    else:
        indexes = np.array(indexes, dtype=np.int32)
    radii = np.asarray(radii, dtype=float)
    n = radii.size
    k = zf.shape[2]
    score = np.zeros((n,k))
    if n == 0:
        return score
    areas = np.square(radii)
    areas *= np.pi
    for ki in range(k):
        zfk=zf[:,:,ki]
        real_score = scipy.ndimage.sum(zfk.real,labels,indexes)
        real_score = fixup_scipy_ndimage_result(real_score)
            
        imag_score = scipy.ndimage.sum(zfk.imag,labels,indexes)
        imag_score = fixup_scipy_ndimage_result(imag_score)
        # one_score = np.sqrt(real_score**2+imag_score**2) / areas
        np.square(real_score, out=real_score)
        np.square(imag_score, out=imag_score)
        one_score = real_score + imag_score
        np.sqrt(one_score, out=one_score)
        one_score /= areas
        score[:,ki] = one_score
    return score

def zernike(zernike_indexes,labels,indexes):
    """Compute the Zernike features for the labels with the label #s in indexes
    
    returns the score per labels and an array of one image per zernike feature
    """
    #
    # "Reverse_indexes" is -1 if a label # is not to be processed. Otherwise
    # reverse_index[label] gives you the index into indexes of the label
    # and other similarly shaped vectors (like the results)
    #
    indexes = np.array(indexes,dtype=np.int32)
    nindexes = len(indexes)
    reverse_indexes = np.empty((np.max(indexes)+1,),int)
    reverse_indexes.fill(-1)
    reverse_indexes[indexes] = np.arange(indexes.shape[0],dtype=int)
    mask = reverse_indexes[labels] != -1

    centers,radii = minimum_enclosing_circle(labels,indexes)
    ny, nx = labels.shape[0:2]
    y, x = np.asarray(np.mgrid[0:ny-1:complex(0,ny),0:nx-1:complex(0,nx)], dtype=float)
    xm = x[mask]
    ym = y[mask]
    lm = labels[mask]
    #
    # The Zernikes are inscribed in circles with points labeled by
    # their fractional distance (-1 <= x,y <= 1) from the center.
    # So we transform x and y by subtracting the center and
    # dividing by the radius
    #
    rev_ind = reverse_indexes[lm]
    ## ym = (ym-centers[reverse_indexes[lm],0]) / radii[reverse_indexes[lm]]
    ym -= centers[rev_ind,0]
    ym /= radii[rev_ind]
    ## xm = (xm-centers[reverse_indexes[lm],1]) / radii[reverse_indexes[lm]]
    xm -= centers[rev_ind,1]
    xm /= radii[rev_ind]
    #
    # Blow up ym and xm into new x and y vectors
    #
    x = np.zeros_like(x)
    x[mask]=xm
    y = np.zeros_like(y)
    y[mask]=ym
    #
    # Pass the resulting x and y through the rest of Zernikeland
    #
    score = np.zeros((nindexes, len(zernike_indexes)))
    zf = construct_zernike_polynomials(x, y, zernike_indexes, mask)
    score = score_zernike(zf, radii, labels, indexes)
    return score

def get_zernike_indexes(limit=10):
    """Return a list of all Zernike indexes up to the given limit
    
    limit - return all Zernike indexes with N less than this limit
    
    returns an array of 2-tuples. Each tuple is organized as (N,M).
    The Zernikes are stored as complex numbers with the real part
    being (N,M) and the imaginary being (N,-M)
    """
    def zernike_indexes_iter(n_max):
        for n in range(0, n_max):
            for m in range(n%2, n+1, 2):
                yield n
                yield m

    z_ind = np.fromiter(zernike_indexes_iter(limit), np.intc)
    z_ind = z_ind.reshape( (len(z_ind) // 2, 2) )
    return z_ind
