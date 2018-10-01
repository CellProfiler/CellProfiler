from __future__ import absolute_import
from __future__ import division
import numpy as np

def otsu(data, min_threshold=None, max_threshold=None,bins=256):
    """Compute a threshold using Otsu's method
    
    data           - an array of intensity values between zero and one
    min_threshold  - only consider thresholds above this minimum value
    max_threshold  - only consider thresholds below this maximum value
    bins           - we bin the data into this many equally-spaced bins, then pick
                     the bin index that optimizes the metric
    """
    assert min_threshold is None or max_threshold is None or min_threshold < max_threshold
    def constrain(threshold):
        if not min_threshold is None and threshold < min_threshold:
            threshold = min_threshold
        if not max_threshold is None and threshold > max_threshold:
            threshold = max_threshold
        return threshold
    
    data = np.atleast_1d(data)
    data = data[~ np.isnan(data)]
    if len(data) == 0:
        return (min_threshold if not min_threshold is None
                else max_threshold if not max_threshold is None
                else 0)
    elif len(data) == 1:
        return constrain(data[0])
    if bins > len(data):
        bins = len(data)
    data.sort()
    var = running_variance(data)
    rvar = np.flipud(running_variance(np.flipud(data))) 
    thresholds = data[1:len(data):len(data)//bins]
    score_low = (var[0:len(data)-1:len(data)//bins] * 
                 np.arange(0,len(data)-1,len(data)//bins))
    score_high = (rvar[1:len(data):len(data)//bins] *
                  (len(data) - np.arange(1,len(data),len(data)//bins)))
    scores = score_low + score_high
    if len(scores) == 0:
        return constrain(thresholds[0])
    index = np.argwhere(scores == scores.min()).flatten()
    if len(index)==0:
        return constrain(thresholds[0])
    #
    # Take the average of the thresholds to either side of
    # the chosen value to get an intermediate in cases where there is
    # a steep step between the background and foreground
    index = index[0]
    if index == 0:
        index_low = 0
    else:
        index_low = index-1
    if index == len(thresholds)-1:
        index_high = len(thresholds)-1
    else:
        index_high = index+1 
    return constrain((thresholds[index_low]+thresholds[index_high]) / 2)

def entropy(data, bins=256):
    """Compute a threshold using Ray's entropy measurement
    
    data           - an array of intensity values between zero and one
    bins           - we bin the data into this many equally-spaced bins, then pick
                     the bin index that optimizes the metric
    """
    
    data = np.atleast_1d(data)
    data = data[~ np.isnan(data)]
    if len(data) == 0:
        return 0
    elif len(data) == 1:
        return data[0]

    if bins > len(data):
        bins = len(data)
    data.sort()
    var = running_variance(data)+1.0/512.0
    rvar = np.flipud(running_variance(np.flipud(data)))+1.0/512.0 
    thresholds = data[1:len(data):len(data)//bins]
    w = np.arange(0,len(data)-1,len(data)//bins)
    score_low = w * np.log(var[0:len(data)-1:len(data)//bins] *
                           w * np.sqrt(2*np.pi*np.exp(1)))
    score_low[np.isnan(score_low)]=0
    
    w = len(data) - np.arange(1,len(data),len(data)//bins)
    score_high = w * np.log(rvar[1:len(data):len(data)//bins] * w *
                            np.sqrt(2*np.pi*np.exp(1)))
    score_high[np.isnan(score_high)]=0
    scores = score_low + score_high
    index = np.argwhere(scores == scores.min()).flatten()
    if len(index)==0:
        return thresholds[0]
    #
    # Take the average of the thresholds to either side of
    # the chosen value to get an intermediate in cases where there is
    # a steep step between the background and foreground
    index = index[0]
    if index == 0:
        index_low = 0
    else:
        index_low = index-1
    if index == len(thresholds)-1:
        index_high = len(thresholds)-1
    else:
        index_high = index+1 
    return (thresholds[index_low]+thresholds[index_high]) / 2

def otsu3(data, min_threshold=None, max_threshold=None,bins=128):    
    """Compute a threshold using a 3-category Otsu-like method
    
    data           - an array of intensity values between zero and one
    min_threshold  - only consider thresholds above this minimum value
    max_threshold  - only consider thresholds below this maximum value
    bins           - we bin the data into this many equally-spaced bins, then pick
                     the bin index that optimizes the metric
    
    We find the maximum weighted variance, breaking the histogram into
    three pieces.
    Returns the lower and upper thresholds
    """
    assert min_threshold is None or max_threshold is None or min_threshold < max_threshold
    
    #
    # Compute the running variance and reverse running variance.
    # 
    data = np.atleast_1d(data)
    data = data[~ np.isnan(data)]
    data.sort()
    if len(data) == 0:
        return 0
    var = running_variance(data)
    rvar = np.flipud(running_variance(np.flipud(data)))
    if bins > len(data):
        bins = len(data)
    bin_len = int(len(data)//bins) 
    thresholds = data[0:len(data):bin_len]
    score_low = (var[0:len(data):bin_len] * 
                 np.arange(0,len(data),bin_len))
    score_high = (rvar[0:len(data):bin_len] *
                  (len(data) - np.arange(0,len(data),bin_len)))
    #
    # Compute the middles
    #
    cs = data.cumsum()
    cs2 = (data**2).cumsum()
    i,j = np.mgrid[0:score_low.shape[0],0:score_high.shape[0]]*bin_len
    diff = (j-i).astype(float)
    w = diff
    mean = (cs[j] - cs[i]) / diff
    mean2 = (cs2[j] - cs2[i]) / diff
    score_middle = w * (mean2 - mean**2)
    score_middle[i >= j] = np.Inf
    score = score_low[i*bins//len(data)] + score_middle + score_high[j*bins//len(data)]
    best_score = np.min(score)
    best_i_j = np.argwhere(score==best_score)
    return (thresholds[best_i_j[0,0]],thresholds[best_i_j[0,1]])

def entropy3(data, bins=128):    
    """Compute a threshold using a 3-category Otsu-like method
    
    data           - an array of intensity values between zero and one
    bins           - we bin the data into this many equally-spaced bins, then pick
                     the bin index that optimizes the metric
    
    We find the maximum weighted variance, breaking the histogram into
    three pieces.
    Returns the lower and upper thresholds
    """
    #
    # Compute the running variance and reverse running variance.
    # 
    data = np.atleast_1d(data)
    data = data[~ np.isnan(data)]
    data.sort()
    if len(data) == 0:
        return 0
    var = running_variance(data)+1.0/512.0
    if bins > len(data):
        bins = len(data)
    bin_len = int(len(data)//bins) 
    thresholds = data[0:len(data):bin_len]
    score_low = entropy_score(var,bins)
    
    rvar = running_variance(np.flipud(data))+1.0/512.0 
    score_high = np.flipud(entropy_score(rvar,bins))
    #
    # Compute the middles
    #
    cs = data.cumsum()
    cs2 = (data**2).cumsum()
    i,j = np.mgrid[0:score_low.shape[0],0:score_high.shape[0]]*bin_len
    diff = (j-i).astype(float)
    w = diff / float(len(data))
    mean = (cs[j] - cs[i]) / diff
    mean2 = (cs2[j] - cs2[i]) / diff
    score_middle = entropy_score(mean2 - mean**2 + 1.0/512.0, bins, w, False)
    score_middle[(i >= j) | np.isnan(score_middle)] = np.Inf
    score = score_low[i//bin_len] + score_middle + score_high[j//bin_len]
    best_score = np.min(score)
    best_i_j = np.argwhere(score==best_score)
    return (thresholds[best_i_j[0,0]],thresholds[best_i_j[0,1]])

def entropy_score(var,bins, w=None, decimate=True):
    '''Compute entropy scores, given a variance and # of bins
    
    '''
    if w is None:
        n = len(var)
        w = np.arange(0,n,n//bins) / float(n)
    if decimate:
        n = len(var)
        var = var[0:n:n//bins]
    score = w * np.log(var * w * np.sqrt(2*np.pi*np.exp(1)))
    score[np.isnan(score)]=np.Inf
    return score
    

def weighted_variance(cs, cs2, lo, hi):
    if hi == lo:
        return np.Infinity
    w = (hi - lo) / float(len(cs))
    mean = (cs[hi] - cs[lo]) / (hi - lo)
    mean2 = (cs2[hi] - cs2[lo]) / (hi - lo)
    return w * (mean2 - mean**2)

def otsu_entropy(cs, cs2, lo, hi):
    if hi == lo:
        return np.Infinity
    w = (hi - lo) / float(len(cs))
    mean = (cs[hi] - cs[lo]) / (hi - lo)
    mean2 = (cs2[hi] - cs2[lo]) / (hi - lo)
    return w * (np.log (w * (mean2 - mean**2) * np.sqrt(2*np.pi*np.exp(1))))

def running_variance(x):
    '''Given a vector x, compute the variance for x[0:i]
    
    Thank you http://www.johndcook.com/standard_deviation.html
    S[i] = S[i-1]+(x[i]-mean[i-1])*(x[i]-mean[i])
    var(i) = S[i] / (i-1)
    '''
    n = len(x)
    # The mean of x[0:i]
    m = x.cumsum() / np.arange(1,n+1)
    # x[i]-mean[i-1] for i=1...
    x_minus_mprev = x[1:]-m[:-1]
    # x[i]-mean[i] for i=1...
    x_minus_m = x[1:]-m[1:]
    # s for i=1...
    s = (x_minus_mprev*x_minus_m).cumsum()
    var = s / np.arange(1,n)
    # Prepend Inf so we have a variance for x[0]
    return np.hstack(([0],var))
