from __future__ import absolute_import
import numpy as np
from scipy.fftpack import fft2
from scipy.ndimage.measurements import sum as nd_sum

def rps(img):
    assert img.ndim == 2
    radii2 = (np.arange(img.shape[0]).reshape((img.shape[0], 1)) ** 2) + (np.arange(img.shape[1])** 2)
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    maxwidth = min(img.shape[0], img.shape[1]) / 8.0 # truncate early to avoid edge effects
    if img.ptp() > 0:
        img = img / np.median(abs(img - img.mean())) # intensity invariant
    mag = abs(fft2(img - np.mean(img)))
    power = mag**2
    radii = np.floor(np.sqrt(radii2)).astype(np.int) + 1
    labels = np.arange(2, np.floor(maxwidth)).astype(np.int).tolist() # skip DC component
    if len(labels) > 0:
        magsum = nd_sum(mag, radii, labels)
        powersum = nd_sum(power, radii, labels)
        return np.array(labels), np.array(magsum), np.array(powersum)
    return [2], [0], [0]
