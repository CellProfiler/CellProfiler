"""zernike.py - compute the zernike moments of an image

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
import scipy.sparse
import scipy.ndimage
from cpmorphology import minimum_enclosing_circle,fixup_scipy_ndimage_result
from cpmorphology import fill_labeled_holes,draw_line

def construct_zernike_lookuptable(zernike_indexes):
    """Return a lookup table of the sum-of-factorial part of the radial
    polynomial of the zernike indexes passed
    
    zernike_indexes - an Nx2 array of the Zernike polynomials to be
                      computed.
    """
    factorial = np.ones((100,))
    factorial[1:] = np.cumproduct(np.arange(1, 100).astype(float))
    width = int(np.max(zernike_indexes[:,0]) / 2+1)
    lut = np.zeros((zernike_indexes.shape[0],width))
    for idx,(n,m) in zip(range(zernike_indexes.shape[0]),zernike_indexes):
        for k in range(0,(n-m)/2+1):
            lut[idx,k] = \
                (((-1)**k) * factorial[n-k] /
                 (factorial[k]*factorial[(n+m)/2-k]*factorial[(n-m)/2-k]))
    return lut

def construct_zernike_polynomials(x,y,zernike_indexes,mask=None):
    """Return the zerike polynomials for all objects in an image
    
    x - the X distance of a point from the center of its object
    y - the Y distance of a point from the center of its object
    zernike_indexes - an Nx2 array of the Zernike polynomials to be computed.
    mask - a mask with same shape as X and Y of the points to consider
    returns a height x width x N array of complex numbers which are the
    e^i portion of the sine and cosine of the Zernikes
    """
    if x.shape != y.shape:
        raise ValueError("X and Y must have the same shape")
    if mask is None:
        mask = np.ones(x.shape,bool)
    elif mask.shape != x.shape:
        raise ValueError("The mask must have the same shape as X and Y")
    x = x[mask]
    y = y[mask]
    lut = construct_zernike_lookuptable(zernike_indexes)
    nzernikes = zernike_indexes.shape[0]
    r = np.sqrt(x**2+y**2)
    phi = np.arctan2(x,y).astype(np.complex)
    zf = np.zeros((x.shape[0], nzernikes), np.complex)
    s = np.zeros(x.shape,np.complex)
    exp_terms = {}
    for idx,(n,m) in zip(range(nzernikes), zernike_indexes):
        s[:]=0
        if not exp_terms.has_key(m):
            exp_terms[m] = np.exp(1j*m*phi)
        exp_term = exp_terms[m]
        for k in range((n-m)/2+1):
            s += lut[idx,k] * r**(n-2*k)
        s[r>1]=0
        zf[:,idx] = s*exp_term 
    
    result = np.zeros((mask.shape[0],mask.shape[1],nzernikes),np.complex)
    result[mask] = zf
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
    radii = np.array(radii)
    k = zf.shape[2]
    n = np.product(radii.shape)
    score = np.zeros((n,k))
    if n == 0:
        return score
    areas = radii**2 * np.pi
    for ki in range(k):
        zfk=zf[:,:,ki]
        real_score = scipy.ndimage.sum(zfk.real,labels,indexes)
        real_score = fixup_scipy_ndimage_result(real_score)
            
        imag_score = scipy.ndimage.sum(zfk.imag,labels,indexes)
        imag_score = fixup_scipy_ndimage_result(imag_score)
        one_score = np.sqrt(real_score**2+imag_score**2) / areas
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
    reverse_indexes = -np.ones((np.max(indexes)+1,),int)
    reverse_indexes[indexes] = np.arange(indexes.shape[0],dtype=int)
    mask = reverse_indexes[labels] != -1

    centers,radii = minimum_enclosing_circle(labels,indexes)
    y,x = np.mgrid[0:labels.shape[0],0:labels.shape[1]]
    xm = x[mask].astype(float)
    ym = y[mask].astype(float)
    lm = labels[mask]
    #
    # The Zernikes are inscribed in circles with points labeled by
    # their fractional distance (-1 <= x,y <= 1) from the center.
    # So we transform x and y by subtracting the center and
    # dividing by the radius
    #
    ym = (ym-centers[reverse_indexes[lm],0]) / radii[reverse_indexes[lm]]
    xm = (xm-centers[reverse_indexes[lm],1]) / radii[reverse_indexes[lm]]
    #
    # Blow up ym and xm into new x and y vectors
    #
    x = np.zeros(x.shape)
    x[mask]=xm
    y = np.zeros(y.shape)
    y[mask]=ym
    #
    # Pass the resulting x and y through the rest of Zernikeland
    #
    score = np.zeros((nindexes, len(zernike_indexes)))
    for i in range(len(zernike_indexes)):
        zf = construct_zernike_polynomials(x, y, zernike_indexes[i:i+1], mask)
        one_score = score_zernike(zf, radii, labels, indexes)
        score[:,i] = one_score[:,0]
    return score

def get_zernike_indexes(limit=10):
    """Return a list of all Zernike indexes up to the given limit
    
    limit - return all Zernike indexes with N less than this limit
    
    returns an array of 2-tuples. Each tuple is organized as (N,M).
    The Zernikes are stored as complex numbers with the real part
    being (N,M) and the imaginary being (N,-M)
    """
    zernike_n_m = []
    for n in range(limit):
        for m in range(n+1):
            if (m+n) & 1 == 0:
                zernike_n_m.append((n,m))
    return np.array(zernike_n_m)

if __name__ == "__main__":
    import wx
    import matplotlib.figure
    import matplotlib.backends.backend_wxagg
    import matplotlib.cm
    import traceback
    
    ORIGINAL_IMAGE = "Original image"
    zernike_indexes = get_zernike_indexes(6)
    y,x = np.mgrid[-100:101,-100:101].astype(float) / 100
    
    zf = construct_zernike_polynomials(x,y, zernike_indexes)
    class MyFrame(wx.Frame):
        def __init__(self):
            wx.Frame.__init__(self, None, title="Zernikes",
                              pos=wx.DefaultPosition, size=wx.DefaultSize,
                              style=wx.DEFAULT_FRAME_STYLE)
            self.figure = matplotlib.figure.Figure()
            self.panel  = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self,-1,self.figure)

            choices = ["%d,%d"%(n,m) for n,m in zernike_indexes]
            self.zcombo = wx.ComboBox(self,-1,value="0,0", choices=choices,
                                      style=wx.CB_DROPDOWN+wx.CB_READONLY)
            sizer = wx.BoxSizer(wx.VERTICAL)
            self.SetSizer(sizer)
            sizer.Add(self.zcombo,0,wx.EXPAND)
            sizer.Add(self.panel,1,wx.EXPAND)
            self.odd_zernike_axes = self.figure.add_subplot(1,3,1)
            self.even_zernike_axes = self.figure.add_subplot(1,3,2)
            self.abs_zernike_axes = self.figure.add_subplot(1,3,3)
            self.imshow(0,0)
            self.Bind(wx.EVT_COMBOBOX,self.on_zernike_change,self.zcombo)
        
        def on_zernike_change(self,event):
            n,m = [int(x) for x in self.zcombo.Value.split(',')]
            self.imshow(n,m)
            self.Refresh()
            
        def imshow(self,n,m):
            for i in range(zernike_indexes.shape[0]):
                if zernike_indexes[i,0]==n and zernike_indexes[i,1] == m:
                    break
            my_zf_even = zf[:,:,i].real
            my_zf_odd = zf[:,:,i].imag
            my_zf_abs = np.abs(zf[:,:,i])
            self.odd_zernike_axes.clear()
            self.odd_zernike_axes.imshow(my_zf_odd, matplotlib.cm.jet)
            self.odd_zernike_axes.set_title("Odd n=%d,m=%d,sum=%f"%(n,m,np.sum(my_zf_odd)))
            self.even_zernike_axes.clear()
            self.even_zernike_axes.imshow(my_zf_even, matplotlib.cm.jet)
            self.even_zernike_axes.set_title("Even n=%d,m=%d,sum=%f"%(n,m,np.sum(my_zf_even)))
            self.abs_zernike_axes.clear()
            self.abs_zernike_axes.imshow(my_zf_abs, matplotlib.cm.jet)
            self.abs_zernike_axes.set_title("Abs n=%d,m=%d,sum=%f"%(n,m,np.sum(my_zf_abs)))
            self.figure.canvas.draw()

    class WildFrame(wx.Frame):
        def __init__(self):
            np.random.seed(0)
            self.labels = self.make_labels()
            self.scores, self.zf = zernike(zernike_indexes, self.labels, 
                                           np.arange(100,dtype=int)+1)
            wx.Frame.__init__(self, None, title="Zernikes",
                              pos=wx.DefaultPosition, size=wx.DefaultSize,
                              style=wx.DEFAULT_FRAME_STYLE)
            self.figure = matplotlib.figure.Figure()
            self.panel  = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(self,-1,self.figure)

            choices = [ORIGINAL_IMAGE]
            choices += ["%d,%d"%(n,m) for n,m in zernike_indexes]
            self.zcombo = wx.ComboBox(self,-1,value="0,0", choices=choices,
                                      style=wx.CB_DROPDOWN+wx.CB_READONLY)
            sizer = wx.BoxSizer(wx.VERTICAL)
            self.SetSizer(sizer)
            sizer.Add(self.zcombo,0,wx.EXPAND)
            sizer.Add(self.panel,1,wx.EXPAND)
            self.odd_zernike_axes = self.figure.add_subplot(1,3,1)
            self.even_zernike_axes = self.figure.add_subplot(1,3,2)
            self.abs_zernike_axes = self.figure.add_subplot(1,3,3)
            self.imshow(None,None)
            self.Bind(wx.EVT_COMBOBOX,self.on_zernike_change,self.zcombo)
        
        def make_labels(self,s=10,side=1000,ct=20):
            mini_side = side / s
            labels = np.zeros((side,side),int)
            pts = np.zeros((s*s*ct,2),int)
            index = np.arange(pts.shape[0],dtype=float)/float(ct)
            index = index.astype(int)
            idx = 0
            for i in range(0,side,mini_side):
                for j in range(0,side,mini_side):
                    idx = idx+1
                    # get ct+1 unique points
                    p = np.random.uniform(low=0,high=mini_side,
                                          size=(ct+1,2)).astype(int)
                    while True:
                        pu = np.unique(p[:,0]+p[:,1]*mini_side)
                        if pu.shape[0] == ct+1:
                            break
                        p[:pu.shape[0],0] = np.mod(pu,mini_side).astype(int)
                        p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                        p_size = (ct+1-pu.shape[0],2)
                        p[pu.shape[0],:] = np.random.uniform(low=0,
                                                             high=mini_side,
                                                             size=p_size)
                    # Use the last point as the "center" and order
                    # all of the other points according to their angles
                    # to this "center"
                    center = p[ct,:]
                    v = p[:ct,:]-center
                    angle = np.arctan2(v[:,0],v[:,1])
                    order = np.lexsort((angle,))
                    p = p[:ct][order]
                    p[:,0] = p[:,0]+i
                    p[:,1] = p[:,1]+j
                    pts[(idx-1)*ct:idx*ct,:]=p
                    #
                    # draw lines on the labels
                    #
                    for k in range(ct):
                        draw_line(labels, p[k,:], p[(k+1)%ct,:], idx)
            labels = fill_labeled_holes(labels)
            return labels
        
        def on_zernike_change(self,event):
            if self.zcombo.Value == ORIGINAL_IMAGE:
                self.imshow(None,None)
            else:
                n,m = [int(x) for x in self.zcombo.Value.split(',')]
                self.imshow(n,m)
            self.Refresh()
            
        def imshow(self,n,m):
            self.odd_zernike_axes.clear()
            self.even_zernike_axes.clear()
            self.abs_zernike_axes.clear()
            if n is None and m is None:
                self.odd_zernike_axes.imshow(self.labels,matplotlib.cm.jet)
                self.odd_zernike_axes.set_title("Original image")
                self.even_zernike_axes.set_visible(False)
                self.abs_zernike_axes.set_visible(False)
                return
            
            self.even_zernike_axes.set_visible(True)
            self.abs_zernike_axes.set_visible(True)
            for i in range(zernike_indexes.shape[0]):
                if zernike_indexes[i,0]==n and zernike_indexes[i,1] == m:
                    break
            my_zf_even = self.zf[:,:,i].real
            my_zf_odd = self.zf[:,:,i].imag
            my_zf_abs = np.abs(self.zf[:,:,i])
            self.odd_zernike_axes.imshow(my_zf_odd, matplotlib.cm.jet)
            self.odd_zernike_axes.set_title("Odd n=%d,m=%d,sum=%f"%(n,m,np.sum(my_zf_odd)))
            self.even_zernike_axes.imshow(my_zf_even, matplotlib.cm.jet)
            self.even_zernike_axes.set_title("Even n=%d,m=%d,sum=%f"%(n,m,np.sum(my_zf_even)))
            self.abs_zernike_axes.imshow(my_zf_abs, matplotlib.cm.jet)
            self.abs_zernike_axes.set_title("Abs n=%d,m=%d,sum=%f"%(n,m,np.sum(my_zf_abs)))
            self.figure.canvas.draw()
    
    
    class MyApp(wx.App):
        def OnInit(self):
            self.frame = MyFrame()
            self.SetTopWindow(self.frame)
            self.frame.Show()
            self.wild_frame = WildFrame()
            self.wild_frame.Show()
            return 1
    app = MyApp(0)
    app.MainLoop()
