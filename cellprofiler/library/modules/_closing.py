import skimage.morphology
from cellprofiler.library.functions.image_processing import morphology_closing


def closing(image, structuring_element, planewise):
    return morphology_closing(
        image,
        structuring_element=structuring_element,
        planewise=planewise,
    )
