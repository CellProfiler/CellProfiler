# from multiprocessing.sharedctypes import Value
import skimage.color
import skimage.morphology
import centrosome


def rgb_to_greyscale(image):
    if image.shape[-1] == 4:
        output = skimage.color.rgba2rgb(image)
        return skimage.color.rgb2gray(output)
    else:
        return skimage.color.rgb2gray(image)


def medial_axis(image):
    if image.ndim > 2 and image.shape[-1] in (3, 4):
        raise ValueError("Convert image to grayscale or use medialaxis module")
    if image.ndim > 2 and image.shape[-1] not in (3, 4):
        raise ValueError("Process 3D images plane-wise or the medialaxis module")
    return skimage.morphology.medial_axis(image)


def enhance_edges_sobel(image, mask=None, direction="all"):
    if direction.casefold() == "all":
        output_pixels = centrosome.filter.sobel(image, mask)
    elif direction.casefold() == "horizontal":
        output_pixels = centrosome.filter.hsobel(image, mask)
    elif direction.casefold() == "vertical":
        output_pixels = centrosome.filter.vsobel(image, mask)
    else:
        raise NotImplementedError(f"Unimplemented direction for Sobel: {direction}")
    return output_pixels


def enhance_edges_log(image, mask=None, sigma=2.0):
    size = int(sigma * 4) + 1
    output_pixels = centrosome.filter.laplacian_of_gaussian(image, mask, size, sigma)
    return output_pixels


def enhance_edges_prewitt(image, mask=None, direction="all"):
    if direction.casefold() == "all":
        output_pixels = centrosome.filter.prewitt(image, mask)
    elif direction.casefold() == "horizontal":
        output_pixels = centrosome.filter.hprewitt(image, mask)
    elif direction.casefold() == "vertical":
        output_pixels = centrosome.filter.vprewitt(image, mask)
    else:
        raise NotImplementedError(f"Unimplemented direction for Prewitt: {direction}")
    return output_pixels


def enhance_edges_canny(
    image,
    mask=None,
    auto_threshold=True,
    auto_low_threshold=True,
    sigma=1.0,
    low_threshold=0.1,
    manual_threshold=0.2,
    threshold_adjustment_factor=1.0,
):

    if auto_threshold or auto_low_threshold:
        sobel_image = centrosome.filter.sobel(image)
        low, high = centrosome.otsu.otsu3(sobel_image[mask])
        if auto_threshold:
            high_th = high * threshold_adjustment_factor
        if auto_low_threshold:
            low_th = low * threshold_adjustment_factor
    else:
        low_th = low_threshold
        high_th = manual_threshold

    output_pixels = centrosome.filter.canny(image, mask, sigma, low_th, high_th)
    return output_pixels
