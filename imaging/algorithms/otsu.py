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

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
import numpy as np
import scipy.ndimage.measurements

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
    thresholds = data[1:len(data):len(data)/bins]
    score_low = (var[0:len(data)-1:len(data)/bins] * 
                 np.arange(0,len(data)-1,len(data)/bins))
    score_high = (rvar[1:len(data):len(data)/bins] *
                  (len(data) - np.arange(1,len(data),len(data)/bins)))
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
    thresholds = data[1:len(data):len(data)/bins]
    w = np.arange(0,len(data)-1,len(data)/bins)
    score_low = w * np.log(var[0:len(data)-1:len(data)/bins] *
                           w * np.sqrt(2*np.pi*np.exp(1)))
    score_low[np.isnan(score_low)]=0
    
    w = len(data) - np.arange(1,len(data),len(data)/bins)
    score_high = w * np.log(rvar[1:len(data):len(data)/bins] * w *
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
    bin_len = int(len(data)/bins) 
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
    score = score_low[i*bins/len(data)] + score_middle + score_high[j*bins/len(data)]
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
    bin_len = int(len(data)/bins) 
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
    score = score_low[i/bin_len] + score_middle + score_high[j/bin_len]
    best_score = np.min(score)
    best_i_j = np.argwhere(score==best_score)
    return (thresholds[best_i_j[0,0]],thresholds[best_i_j[0,1]])

def entropy_score(var,bins, w=None, decimate=True):
    '''Compute entropy scores, given a variance and # of bins
    
    '''
    if w is None:
        n = len(var)
        w = np.arange(0,n,n/bins) / float(n)
    if decimate:
        n = len(var)
        var = var[0:n:n/bins]
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
    var = s / np.arange(2,n+1)
    # Prepend Inf so we have a variance for x[0]
    return np.hstack(([0],var))
    
    
if __name__=='__main__':
    import PIL.Image as PILImage
    import wx
    import os
    from matplotlib.image import pil_to_array
    import cellprofiler.gui.cpfigure as F
    from cellprofiler.cpmath.filter import stretch
    from cellprofiler.cpmath.threshold import log_transform, inverse_log_transform
    
    FILE_OPEN = wx.NewId()
    M_OTSU = wx.NewId()
    M_ENTROPY = wx.NewId()
    M_OTSU3 = wx.NewId()
    M_ENTROPY3 = wx.NewId()
    M_FAST_OTSU3 = wx.NewId()
    M_LOG_TRANSFORM = wx.NewId()
    class MyApp(wx.App):
        def OnInit(self):
            self.frame = F.CPFigureFrame(title="Otsu",subplots=(2,1))
            file_menu = self.frame.MenuBar.Menus[0][0]
            file_menu.Append(FILE_OPEN,"&Open")
            file_menu.AppendRadioItem(M_OTSU,"Otsu")
            file_menu.AppendRadioItem(M_ENTROPY,"Entropy")
            file_menu.AppendRadioItem(M_OTSU3,"Otsu3")
            file_menu.AppendRadioItem(M_ENTROPY3, "Entropy3")
            file_menu.AppendRadioItem(M_FAST_OTSU3,"Otsu3 Fast & Messy")
            file_menu.AppendCheckItem(M_LOG_TRANSFORM, "Log transform")
            wx.EVT_MENU(self.frame,FILE_OPEN,self.on_file_open)
            self.SetTopWindow(self.frame)
            self.frame.Show()
            return 1
        
        def on_file_open(self, event):
            dlg = wx.FileDialog(self.frame,style=wx.FD_OPEN)
            if dlg.ShowModal() == wx.ID_OK:
                img = pil_to_array(PILImage.open(os.path.join(dlg.GetDirectory(),dlg.GetFilename())))
                if img.ndim == 3:
                    img = img[:,:,0]+img[:,:,1]+img[:,:,2]
                img = stretch(img.astype(float))
                lt = self.frame.MenuBar.Menus[0][0].MenuItems[7].IsChecked()
                if lt:
                    limg, d = log_transform(img)
                else:
                    limg = img
                self.frame.subplot_imshow_grayscale(0, 0, limg)
                limg = limg.flatten()
                menu_items = self.frame.MenuBar.Menus[0][0].MenuItems
                if menu_items[2].IsChecked():
                    t1 = t2 = otsu(limg)
                elif menu_items[3].IsChecked():
                    t1 = t2 = entropy(limg)
                elif menu_items[4].IsChecked():
                    t1, t2 = otsu3slow(limg)
                elif menu_items[5].IsChecked():
                    t1, t2 = entropy3(limg)
                else:
                    t1, t2 = otsu3(limg)
                if lt:
                    t1,t2 = inverse_log_transform(np.array([t1,t2]), d)
                m1 = img < t1
                m2 = np.logical_and(img >= t1, img < t2)
                m3 = img > t2
                cimg = np.zeros((m1.shape[0],m1.shape[1],3))
                cimg[:,:,0][m1]=img[m1]
                cimg[:,:,1][m2]=img[m2]
                cimg[:,:,2][m3]=img[m3]
                self.frame.subplot_imshow(1, 0, cimg,
                                          sharex = self.frame.subplot(0,0),
                                          sharey = self.frame.subplot(0,0))
                self.frame.Refresh()
                wx.MessageBox("Low threshold = %f, high threshold = %f"%(t1,t2),
                              parent=self.frame)
    app = MyApp(0)
    app.MainLoop()

     

    
    
