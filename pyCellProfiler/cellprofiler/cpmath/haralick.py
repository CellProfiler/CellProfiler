# Not tested -- at all!

import scipy.ndimage as scind
from scipy.linalg.basic import toeplitz

def haralick(image, labels, scale):
    """
    Calculate the Haralick texture features.

    image -- 2-D numpy array of 32-bit floating-point numbers.
    labels -- 2-D numpy array of integers.
    scale -- an integer.

    Returns a (n x 13) numpy array of floating-point numbers,
    where n is the number of objects.
    """
    
    # Normalize the intensities of each object to the [0, 1] range.
    nobjects = labels.max()
    lmax = numpy.array(scind.maximum(image, labels, range(nobjects + 1)))
    lmin = numpy.array(scind.minimum(image, labels, range(nobjects + 1)))
    # Divisor is the object's max - min, or 1 if they are the same.
    divisor = numpy.ones((nobjects + 1,))
    divisor[lmax > lmin] = (lmax - lmin)[lmax > lmin]
    normalized = (image - lmin[labels]) / divisor[labels]

    # Quantize the normalized intensities.
    nlevels = 8
    quantized = numpy.array(normalized // (1.0 / nlevels), dtype='i1')
    quantized = quantized.clip(0, nlevels - 1)

    # Calculate the co-occurrence matrix P.  Given a row number I in
    # the matrix, let A be the set of pixels with that gray level,
    # excluding pixels in the rightmost S columns of the image.  Let B
    # be the set of pixels S pixels to the right of A in the image.
    # Row I is a gray-level histogram of the pixels in B.
    image_a = quantized[:, :-scale]
    image_b = quantized[:, scale:]
    P = numpy.array([numpy.histogram(image_b[image_a == i], new=True)[0]
                     for i in range(nlevels)])

    px = P.sum(1)
    py = P.sum(0)
    r = numpy.arange(levels)
    mux = (r + 1) * px
    muy = (r + 1) * py
    sigmax = numpy.sqrt(numpy.sum((r + 1 - mux) ** 2 * px))
    sigmay = numpy.sqrt(numpy.sum((r + 1- muy) ** 2 * py))
    eps = numpy.finfo(float).eps
    hx = -numpy.sum(px * numpy.log(px + eps))
    hy = -numpy.sum(py * numpy.log(py + eps))
    hxy = -numpy.sum(P.flatten() * numpy.log(P.flatten() + eps))
    pxpy = numpy.mat(px).T * numpy.mat(py)
    hxy1 = -numpy.sum(P * numpy.log(pxpy + eps))
    hxy2 = -numpy.sum(pxpy * numpy.log(pxpy + eps))

    p_xplusy = numpy.zeros((2 * levels - 1, 1))
    p_xminusy = numpy.zeros((levels, 1))
    for x in r:
        for y in r:
            p_xplusy[x + y] += P[x, y]
            p_xminusy[numpy.abs(x - y)] += P[x, y]

    # H1.  Angular second moment.
    H1 = (P.flatten() ** 2).sum()
    
    # H2.  Contrast.
    H2 = (r ** 2 * p_xminusy).sum()

    # H3.  Correlation.
    H3 = numpy.multiply(numpy.mat(r) * numpy.mat(r), P)
    H3[H3 == numpy.inf] = 0

    # H4.  Sum of squares: variation.
    H4 = sigmax ** 2
    
    # H5.  Inverse difference moment.
    H5 = (1 / (1 + toeplitz(r) ** 2) * P).sum()

    # H6.  Sum average.
    H6 = numpy.sum((r + 1) * 2 * p_xplusy)

    # H7.  Sum variance (error in Haralick's original paper here).
    H7 = numpy.sum(((r + 1) * 2 - H6) ** 2 * p_xplusy)

    # H8.  Sum entropy.
    H8 = -numpy.sum(p_xplusy * numpy.log(p_xplusy + eps))

    # H9.  Entropy.
    H9 = hxy
    
    # H10.  Difference variance.
    H10 = numpy.sum(p_xminusy * (r - 1 - numpy.sum((r - 1) * p_xminusy)) ** 2)

    # H11.  Difference entropy.
    H11 = -numpy.sum(p_xminusy * numpy.log(p_xminusy + eps))

    # H12.  Information measure of correlation 1.
    H12 = (hxy - hxy1) / max(hx, hy)
    
    # H13.  Information measure of correlation 2.
    H13 = numpy.sqrt(1 - numpy.exp(-2 * (hxy2 - hxy)))
    # An imaginary result has been encountered once, reason unclear.
    
    return [H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13]


if __name__ == '__main__':
    pass
