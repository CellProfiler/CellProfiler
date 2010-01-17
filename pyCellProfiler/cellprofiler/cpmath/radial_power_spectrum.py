import numpy as np
from scipy.fftpack import fft2
from scipy.sparse import lil_matrix
from scipy.ndimage.measurements import sum as nd_sum

def rps(img):
    assert img.ndim == 2
    radii2 = (np.arange(img.shape[0]).reshape((img.shape[0], 1)) ** 2) + (np.arange(img.shape[1])** 2)
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    halfwidth = min(img.shape[0], img.shape[1]) / 4.0 # truncate early to avoid ringing
    if img.ptp() > 0:
        img = img / np.median(abs(img - img.mean())) # intensity invariant
    rfft = abs(fft2(img - np.mean(img)))**2
    radii = np.floor(np.sqrt(radii2)).astype(np.int) + 1
    labels = np.arange(2, np.floor(halfwidth)).astype(np.int).tolist() # skip DC component
    if len(labels) > 0:
        sums = nd_sum(rfft, radii, labels)
        return np.array(labels), np.array(sums)
    return [2], [0]
