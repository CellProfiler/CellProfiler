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
    assert np.min(data) >= 0, "The input data must be greater than zero"
    assert np.max(data) <= 1, "The input data must be less than or equal to one"
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
        cT = float(np.sum(int_data))        # the count: # of pixels in array
        c0 = float(np.sum(int_data[:k]))    # the # of pixels in the lower group
        c1 = float(np.sum(int_data[k:]))    # the # of pixels in the upper group
        if c0 == 0 or c1 == 0:
            continue
        w0 = c0 / cT                           # the probability of a pixel being in the lower group
        w1 = c1 / cT                           # the probability of a pixel being in the upper group
        r0 = np.array(range(0,k),dtype=float)    # 0 to k-1 as floats
        r1 = np.array(range(k,bins),dtype=float) # k to bins-1 as floats
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
    assert np.min(data) >= 0, "The input data must be greater than zero"
    assert np.max(data) <= 1, "The input data must be less than or equal to one"
    assert min_threshold==None or min_threshold >=0
    assert min_threshold==None or min_threshold <=1
    assert max_threshold==None or max_threshold >=0
    assert max_threshold==None or max_threshold <=1
    assert min_threshold==None or max_threshold==None or min_threshold < max_threshold
    
    v=data[np.logical_and(np.logical_or(min_threshold==None, 
                                        data >= min_threshold),
                          np.logical_or(max_threshold==None,
                                        data <= max_threshold))]
    if len(v) == 0:
        return (0,1)
    v.sort()
    cs = v.cumsum()
    cs2 = (v**2).cumsum()
    thresholds = v[::len(v)/bins]
    best = np.inf
    max_k = [(0,1)]
    skip = len(v) / bins
    for tlo in range(1, len(v), skip):
        for thi in range(tlo + len(v) / bins, len(v), skip):
            score = (weighted_variance(cs, cs2, 0, tlo) + 
                     weighted_variance(cs, cs2, tlo, thi) + 
                     weighted_variance(cs, cs2, thi, len(v) - 1))
            if score < best:
                best = score
                max_k = [(tlo,thi)]
            elif score == best:
                max_k.append((tlo,thi))
    #
    # Find the longest consecutive run of k0 and k1 and take the midpoint
    # of that run
    #
    max_k = np.array(max_k)
    if max_k.ndim == 1:
        return (v[max_k[0]],v[max_k[1]])
    best_k = np.zeros((2,))
    for j in range(0,2):
        run_length = 0
        best_run_length = 0
        for i in range(max_k.shape[0]):
            run_length += 1
            if i == max_k.shape[0]-1 or max_k[i,j]+skip != max_k[i+1,j]:
                if run_length > best_run_length:
                    best_k[j] = int((max_k[i,j]+max_k[i-run_length+1,j]) / 2)
                    best_run_length = run_length
                run_length = 0
    return (v[best_k[0]],v[best_k[1]])

def weighted_variance(cs, cs2, lo, hi):
    w = (hi - lo) / float(len(cs))
    mean = (cs[hi] - cs[lo]) / (hi - lo)
    mean2 = (cs2[hi] - cs2[lo]) / (hi - lo)
    return w * (mean2 - mean**2)

if __name__=='__main__':
    import PIL.Image
    import wx
    import os
    from matplotlib.image import pil_to_array
    import cellprofiler.gui.cpfigure as F
    from cellprofiler.cpmath.filter import stretch
    
    FILE_OPEN = wx.NewId()
    class MyApp(wx.App):
        def OnInit(self):
            wx.InitAllImageHandlers()
            self.frame = F.CPFigureFrame(title="Otsu",subplots=(2,1))
            file_menu = self.frame.MenuBar.Menus[0][0]
            file_menu.Append(FILE_OPEN,"&Open")
            wx.EVT_MENU(self.frame,FILE_OPEN,self.on_file_open)
            self.SetTopWindow(self.frame)
            self.frame.Show()
            return 1
        
        def on_file_open(self, event):
            dlg = wx.FileDialog(self.frame,style=wx.FD_OPEN)
            if dlg.ShowModal() == wx.ID_OK:
                img = pil_to_array(PIL.Image.open(os.path.join(dlg.GetDirectory(),dlg.GetFilename())))
                if img.ndim == 3:
                    img = img[:,:,0]+img[:,:,1]+img[:,:,2]
                img = stretch(img.astype(float))
                self.frame.subplot_imshow_grayscale(0, 0, img)
                t1, t2 = otsu3(img.flat)
                m1 = img < t1
                m2 = np.logical_and(img >= t1, img < t2)
                m3 = img > t2
                cimg = np.zeros((m1.shape[0],m1.shape[1],3))
                cimg[:,:,0][m1]=img[m1]
                cimg[:,:,1][m2]=img[m2]
                cimg[:,:,2][m3]=img[m3]
                self.frame.subplot_imshow_color(1, 0, cimg)
                self.frame.Refresh()
                wx.MessageBox("Low threshold = %f, high threshold = %f"%(t1,t2),
                              parent=self.frame)
    app = MyApp(0)
    app.MainLoop()

     

    
    
