from __future__ import absolute_import
from __future__ import print_function
import warnings

import numpy as np
from scipy import linspace
from scipy.ndimage import affine_transform
from six.moves import range

'''Automatically determine whether background is darker than foreground'''
MODE_AUTO = "auto"

'''Background is darker than foreground'''
MODE_DARK = "dark"

'''Background is brighter than foreground'''
MODE_BRIGHT = "bright"

'''Some foreground is darker, some is lighter'''
MODE_GRAY = "gray"

def prcntiles(x,percents):
    '''Equivalent to matlab prctile(x,p), uses linear interpolation.'''
    x=np.array(x).flatten()
    listx = np.sort(x)
    xpcts=[]
    lenlistx=len(listx)
    refs=[]
    for i in range(0,lenlistx):
        r=100*((.5+i)/lenlistx) #refs[i] is percentile of listx[i] in matrix x
        refs.append(r)

    rpcts=[]
    for p in percents:
        if p<refs[0]:
            rpcts.append(listx[0])
        elif p>refs[-1]:
            rpcts.append(listx[-1])
        else:
            for j in range(0,lenlistx): #lenlistx=len(refs)
                if refs[j]<=p and refs[j+1]>=p:
                    my=listx[j+1]-listx[j]
                    mx=refs[j+1]-refs[j]
                    m=my/mx #slope of line between points
                    rpcts.append((m*(p-refs[j]))+listx[j])
                    break
    xpcts.append(rpcts)
    return np.array(xpcts).transpose()

    
def automode(data):
    '''Tries to guess if the image contains dark objects on a bright background (1)
or if the image contains bright objects on a dark background (-1),
or if it contains both dark and bright objects on a gray background (0).'''

    
    
    pct=prcntiles(np.array(data),[1,20,80,99])
    
    upper=pct[3]-pct[2]
    mid=pct[2]-pct[1]
    lower=pct[1]-pct[0]

    ##print 'upper = '+str(upper)
    ##print 'mid = '+str(mid)
    ##print 'lower = '+str(lower)

    #upper objects
    if upper>mid:
        uo=1
    else:
        uo=0
    ##print 'uo = '+str(uo)

    #lower objects
    if lower>mid:
        lo=1
    else:
        lo=0
    ##print 'lo = '+str(lo)
    
    if uo==1:
        if lo==1:
            mode=0
            #both upper and lower objects
        else:
            mode=-1
            #only upper objects
    else:
        if lo==1:
            mode=1
            #only lower objects
        else:
            mode=0
            #no objects at all
            
    return mode

def spline_factors(u):
    '''u is np.array'''

    X = np.array([(1.-u)**3 , 4-(6.*(u**2))+(3.*(u**3)) , 1.+(3.*u)+(3.*(u**2))-(3.*(u**3)) , u**3]) * (1./6)
    
    return X

def pick(picklist,val):
    '''Index to first value in picklist that is larger than val.
If none is larger, index=len(picklist).'''
    
    assert np.all(np.sort(picklist) == picklist), "pick list is not ordered correctly"
    val = np.array(val)
    i_pick, i_val = np.mgrid[0:len(picklist),0:len(val)]
    #
    # Mark a picklist entry as 1 if the value is before or at,
    # mark it as zero if it is afterward
    #
    is_not_larger = picklist[i_pick] <= val[i_val]
    #
    # The index is the number of entries that are 1
    #
    p = np.sum(is_not_larger, 0)
    
    return p
        
def confine(x,low,high):
    '''Confine x to [low,high]. Values outside are set to low/high.
See also restrict.'''

    y=x.copy()
    y[y < low] = low
    y[y > high] = high
    return y

def gauss(x,m_y,sigma):
    '''returns the gaussian with mean m_y and std. dev. sigma,
calculated at the points of x.'''

    e_y = [np.exp((1.0/(2*float(sigma)**2)*-(n-m_y)**2)) for n in np.array(x)]
    y = [1.0/(float(sigma) * np.sqrt(2 * np.pi)) * e for e in e_y]
    
    return np.array(y)

def d2gauss(x,m_y,sigma):
    '''returns the second derivative of the gaussian with mean m_y,
and standard deviation sigma, calculated at the points of x.'''

    return gauss(x,m_y,sigma)*[-1/sigma**2 + (n-m_y)**2/sigma**4 for n in x]

def spline_matrix(x,px):
    n=len(px)
    lx=len(x)

    # Assign each x to an interval.  Subtract 1 to get the beginning of the interval.
    px = np.array(px)
    x = np.array(x)
    j = np.array(pick(px,x)) - 1
    #
    # We need at least one entry before and two after for the four factors
    # of the cubic spline
    #
    j = confine(j, 1, n-3)

    u = (x-px[j]) / (px[j+1]-px[j]) #how far are we on the line segment px[j]->px[j+1], 0<=u<1

    spf=spline_factors(u)
    #
    # Set up to broadcast spf to the correct spline factors
    # The cubic has four factors that broadcast starting at j-1 to j+2 
    #
    ii, jj = np.mgrid[0:spf.shape[0], 0:lx]
    V = np.zeros((n, lx))
    V[j[jj] - 1 +  ii, jj] = spf[ii, jj]
    return V

def spline_matrix2d(x,y,px,py,mask=None):
    '''For boundary constraints, the first two and last two spline pieces are constrained
to be part of the same cubic curve.'''
    V = np.kron(spline_matrix(x,px),spline_matrix(y,py))
    
    lenV = len(V)
    
    if mask is not None:
        indices = np.nonzero(mask.T.flatten())
        if len(indices)>1:
            indices = np.nonzero(mask.T.flatten())[1][0]
        newV=V.T[indices]
        V=newV.T
        
        V=V.reshape((V.shape[0],V.shape[1]))
        
    return V


def splinefit2d(x, y, z, px, py, mask=None):
    '''Make a least squares fit of the spline (px,py,pz) to the surface (x,y,z).
If mask is given, only masked points are used for the regression.'''
    
    if mask is None:
        V = np.array(spline_matrix2d(x, y, px, py))
        a = np.array(z.T.flatten())
        pz = np.linalg.lstsq(V.T, a.T)[0].T
    else:
        V = np.array(spline_matrix2d(x,y,px,py,mask))
        indices = np.nonzero(np.array(mask).T.flatten())
        if len(indices[0])==0:
            pz = np.zeros((len(py),len(px)))
        #indices is empty when mask changes to all zeros
        else:
            a = np.array((z.T.flatten()[indices[0]]))
            pz = np.linalg.lstsq(V.T, a.T)[0].T
            
    pz=pz.reshape((len(py),len(px)))
    return pz.transpose()

def splineimage(Z, points, mask=None, x=None, y=None):
    
    [r,c]=Z.shape
    if x is None:
        x = np.arange(c)
    if y is None:
        y = np.arange(r)

    #spline control points, let the outer ones be outside the image
    cstep = (x[-1] - x[0]) / (points - 3.)
    px = linspace(x[0] - cstep, x[-1] + cstep, points)
    rstep = (y[-1] - y[0]) / (points - 3.)
    py = linspace(y[0] - rstep, y[-1] + rstep, points)

    if mask is None:
        pz = splinefit2d(x, y, Z, px, py)
    else:
        pz = splinefit2d(x, y, Z, px, py, mask)
    return px, py, pz

def evalspline2d(x, y, px, py, pz):

    V = spline_matrix2d(x,y,px,py)
    a = np.dot(pz.T.reshape((1, np.prod(pz.shape))),V)
    z = a.reshape((len(x),len(y))).T
    
    return z
    
def unbiased_std(a):
    return np.sqrt(((a - np.mean(a)) ** 2).sum() / (len(a) - 1))

def backgr(img, mask = None, mode=MODE_AUTO, thresh=2, splinepoints=5, scale=1, 
           maxiter=40, convergence = .001):
    '''Iterative spline-based background correction.
    
    mode - one of MODE_AUTO, MODE_DARK, MODE_BRIGHT or MODE_GRAY
    thresh - thresh is threshold to cut at, in units of sigma. 
    splinepoints - # of points in spline in each direction
    scale - scale the image by this factor (e.g. 2 = operate on 1/2 of the points)
    maxiter - maximum # of iterations
    convergence - result has converged when the standard deviation of the
                  difference between iterations is this fraction of the
                  maximum image intensity.
Spline mesh is splinepoints x splinepoints. Modes are
defined by the background intensity, Larger thresh->slower but
more stable convergence. Returns background matrix.'''

    assert img.ndim == 2, "Image must be 2-d"
    assert splinepoints >= 3, "The minimum grid size is 3x3"
    assert maxiter >= 1
    assert mode in [MODE_AUTO, MODE_BRIGHT, MODE_DARK, MODE_GRAY], mode + " is not a valid background mode"
    orig_shape = np.array(img.shape).copy()
    input_mask = mask
    if mask is None:
        mask = np.ones(orig_shape, dtype=bool) #start with mask = whole image
        clip_imin = clip_jmin = 0
        clip_imax = img.shape[0]
        clip_jmax = img.shape[1]
        clip_shape = orig_shape
    else:
        isum = np.sum(mask, 1)
        jsum = np.sum(mask, 0)
        clip_imin = np.min(np.argwhere(isum != 0))
        clip_imax = np.max(np.argwhere(isum != 0)) + 1
        clip_jmin = np.min(np.argwhere(jsum != 0))
        clip_jmax = np.max(np.argwhere(jsum != 0)) + 1
        clip_shape = np.array([clip_imax - clip_imin, clip_jmax - clip_jmin])
        
    subsample_shape = (clip_shape / scale).astype(int)
    ratio = (clip_shape.astype(float) - 1) / (subsample_shape.astype(float) - 1)
    transform = np.array([[ratio[0], 0], [0, ratio[1]]])
    inverse_transform = np.array([[1.0 / ratio[0], 0],
                                  [0, 1.0 / ratio[1]]])
    
    img = affine_transform(img[clip_imin:clip_imax, clip_jmin:clip_jmax],
                           transform, output_shape=tuple(subsample_shape),
                           order = 2)
    mask = affine_transform(mask[clip_imin:clip_imax, 
                                 clip_jmin:clip_jmax].astype(float), 
                            transform, 
                            output_shape=tuple(subsample_shape),
                            order = 2) > .5
    orig_mask = mask
    
    if mode=='auto':
        mode = automode(img[orig_mask])
    elif mode=='dark' or mode=='low':
        mode = -1
    elif mode=='bright' or mode=='high':
        mode = 1
    elif mode=='gray' or mode=='grey' or mode=='mid':
        mode = 0

    # Base the stop criterion on a fraction of the image dynamic range
    
    stop_criterion=max((np.max(img) - np.min(img)) * convergence,
                       np.finfo(img.dtype).eps)
    [r,c] = img.shape

    oldres = np.zeros((r,c)) #old background

    for i in range(maxiter):
        px, py, pz = splineimage(img, splinepoints, np.array(mask)) #now with mask
        res = evalspline2d(np.arange(c), np.arange(r), px, py, pz)
        comp = img - res

        diff = res[orig_mask] - oldres[orig_mask]
        ###Compute std. deviation in same way as matlab std(), (not numpy.std())
        stddiff = unbiased_std(diff)

        if stddiff < stop_criterion:#stop_criterion instead of .004
            break
        elif i==maxiter:
            warnings.warn('Background did not converge after %d iterations.\nMake sure that the foreground/background mode is correct.'%(i))

        oldres = res
        
        #calculate new mask
        backgr = comp[mask]
        sigma = unbiased_std(backgr)
        cut = sigma * thresh
        
        if mode < 0:
            mask = comp < cut
        elif mode > 0:
            mask = comp > -cut
        else:
            mask = abs(comp) < cut
        mask &= orig_mask
        nnz = np.sum(mask)
        if nnz < .01 * np.sum(orig_mask):
            warnings.warn('Less than 1%% of the pixels used for fitting,\ntry starting again with a larger threshold value')
            break
    
    output = np.zeros(orig_shape, img.dtype)
    output[clip_imin:clip_imax, clip_jmin:clip_jmax] = \
          affine_transform(res, inverse_transform, 
                           output_shape = tuple(clip_shape),
                           order = 3)
    if input_mask is not None:
        output[~input_mask] = 0
    return output

def bg_compensate(img, sigma, splinepoints, scale):
    '''Reads file, subtracts background. Returns [compensated image, background].'''
    
    from PIL import Image
    import pylab
    from matplotlib.image import pil_to_array
    from centrosome.filter import canny
    import matplotlib

    img = Image.open(img)
    if img.mode=='I;16':
        # 16-bit image
        # deal with the endianness explicitly... I'm not sure
        # why PIL doesn't get this right.
        imgdata = np.fromstring(img.tostring(),np.uint8)
        imgdata.shape=(int(imgdata.shape[0]/2),2)
        imgdata = imgdata.astype(np.uint16)
        hi,lo = (0,1) if img.tag.prefix == 'MM' else (1,0)
        imgdata = imgdata[:,hi]*256 + imgdata[:,lo]
        img_size = list(img.size)
        img_size.reverse()
        new_img = imgdata.reshape(img_size)
        # The magic # for maximum sample value is 281
        if 281 in img.tag:
            img = new_img.astype(np.float32) / img.tag[281][0]
        elif np.max(new_img) < 4096:
            img = new_img.astype(np.float32) / 4095.
        else:
            img = new_img.astype(np.float32) / 65535.
    else:
        img = pil_to_array(img)
    
    pylab.subplot(1,3,1).imshow(img, cmap=matplotlib.cm.Greys_r)
    pylab.show()
    
    if len(img.shape)>2:
        raise ValueError('Image must be grayscale')

## Create mask that will fix problem when image has black areas outside of well
    edges = canny(img, np.ones(img.shape, bool), 2, .1, .3)
    ci = np.cumsum(edges, 0)
    cj = np.cumsum(edges, 1)
    i,j = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    mask = ci > 0
    mask = mask & (cj > 0)
    mask[1:,:] &= (ci[0:-1,:] < ci[-1,j[0:-1,:]])
    mask[:,1:] &= (cj[:,0:-1] < cj[i[:,0:-1],-1])
    
    import time
    t0 = time.clock()
    bg = backgr(img, mask, MODE_AUTO, sigma, splinepoints=splinepoints, scale=scale)
    print("Executed in %f sec" % (time.clock() - t0))
    bg[~mask] = img[~mask]

    pylab.subplot(1,3,2).imshow(img - bg, cmap=matplotlib.cm.Greys_r)
    pylab.subplot(1,3,3).imshow(bg, cmap=matplotlib.cm.Greys_r)
    pylab.show()
