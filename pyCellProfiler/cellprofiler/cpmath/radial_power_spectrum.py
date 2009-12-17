import numpy as np
from scipy.fftpack import fft2
from scipy.sparse import lil_matrix
from scipy.ndimage.measurements import sum as nd_mean

def rps(img):
    assert img.ndim == 2
    rfft = abs(fft2(img - np.mean(img))) / np.prod(img.shape)
    radii2 = (np.arange(img.shape[0]).reshape((img.shape[0], 1)) ** 2) + (np.arange(img.shape[1])** 2)
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    labels = list(set((radii2 + 1).flat))
    labels.sort()
    means = nd_mean(rfft, radii2 + 1, labels)
    return np.sqrt(np.array(labels)), means

