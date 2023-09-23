from ..functions.image_processing import morphology_closing


def closing(image, structuring_element):
    return morphology_closing(
        image,
        structuring_element=structuring_element,
    )
