from cellprofiler_library.functions.image_processing import gaussian_filter

def gaussianfilter(image, sigma, perZ_3D = False):
    return gaussian_filter(
        image,
        sigma,
    )