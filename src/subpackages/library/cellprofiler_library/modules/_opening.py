from ..functions.image_processing import morphology_opening

def opening(image, structuring_element):
    return morphology_opening(
        image,
        structuring_element,
    )