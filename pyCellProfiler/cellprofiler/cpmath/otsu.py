"""Otsu's method:

Otsu's method (N. Otsu, "A Threshold Selection Method from Gray-Level Histograms", 
IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62-66, 1979.)

Consider the sets of pixels in C classified as being above and below a threshold k: C0=I(I<=k) and C1=I(I>k)
    * The probability of a pixel being in C0, C1 or C is w0,w1,or wT(=1), 
      the mean values of the classes are u0, u1 and uT 
      and the variances are s0,s1,sT
    * Define within-class variance and between-class variance as 
      sw=w0*s0+w1*s1 and sb=w0(u0-uT)^2 +w1(u1-uT)^2 = w0*w1(u1-u0)^2
    * Define L = sb/sw, K = sT / sw and N=sb/sT
    * Well-thresholded classes should be separated in gray levels, so sb should
      be maximized wrt to sw, sT wrt to sw and sb wrt to sT. It turns out that
      satisfying any one of these satisfies the others.
    * sT is independent of choice of threshold k, so N can be maximized with 
      respect to K by maximizing sb.
    * Algorithm: compute w0*w1(u1-u0)^2 for all k and pick the maximum. 

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import numpy
import scipy.ndimage.measurements

def otsu(data, min_threshold=None, max_threshold=None,bins=256):
    """Compute a threshold using Otsu's method
    
    data           - an array of intensity values between zero and one
    min_threshold  - only consider thresholds above this minimum value
    max_threshold  - only consider thresholds below this maximum value
    bins           - we bin the data into this many equally-spaced bins, then pick
                     the bin index that optimizes the metric
    """
    assert numpy.min(data) >= 0, "The input data must be greater than zero"
    assert numpy.max(data) <= 1, "The input data must be less than or equal to one"
    assert min_threshold==None or min_threshold >=0
    assert min_threshold==None or min_threshold <=1
    assert max_threshold==None or max_threshold >=0
    assert max_threshold==None or max_threshold <=1
    assert min_threshold==None or max_threshold==None or min_threshold < max_threshold
    
    int_data = scipy.ndimage.measurements.histogram(data,0,1,bins)
    min_bin = (min_threshold and (int(bins * min_threshold)+1)) or 1
    max_bin = (max_threshold and (int(bins * max_threshold)-1)) or (bins-1)
    max_score     = 0
    max_k         = min_bin
    n_max_k       = 0                          # # of k in a row at max
    last_was_max  = False                      # True if last k was max 
    for k in range(min_bin,max_bin):
        cT = float(numpy.sum(int_data))        # the count: # of pixels in array
        c0 = float(numpy.sum(int_data[:k]))    # the # of pixels in the lower group
        c1 = float(numpy.sum(int_data[k:]))    # the # of pixels in the upper group
        if c0 == 0 or c1 == 0:
            continue
        w0 = c0 / cT                           # the probability of a pixel being in the lower group
        w1 = c1 / cT                           # the probability of a pixel being in the upper group
        r0 = numpy.array(range(0,k),dtype=float)    # 0 to k-1 as floats
        r1 = numpy.array(range(k,bins),dtype=float) # k to bins-1 as floats
        u0 = sum(int_data[:k]*r0) / c0              # the average value in the lower group
        u1 = sum(int_data[k:]*r1) / c1              # the average value in the upper group
        score = w0*w1*(u1-u0)*(u1-u0)
        if score > max_score:
            max_k = k
            max_score = score
            n_max_k = 1
            last_was_max = True
        elif score == max_score and last_was_max:
            max_k   += k
            n_max_k += 1
        elif last_was_max:
            last_was_max = False
    if n_max_k == 0:
        max_k = min_bin+max_bin-1
        n_max_k = 2
    return float(max_k) / float(bins * n_max_k) 

    
