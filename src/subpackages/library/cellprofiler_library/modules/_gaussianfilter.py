from cellprofiler_library.functions.image_processing import gaussian_filter

def gaussianfilter(image, sigma):
    return gaussian_filter(
        image,
        sigma,
    )