from cellprofiler.library.functions.image_processing import rgb_to_greyscale, medial_axis
import numpy

def medialaxis(image, multichannel, volumetric):
    if multichannel:
        image = rgb_to_greyscale(image)

    if volumetric:
        data = numpy.zeros_like(image)

        for z, plane in enumerate(image):
            data[z] = medial_axis(plane)
        return data
    else:
        return medial_axis(image)