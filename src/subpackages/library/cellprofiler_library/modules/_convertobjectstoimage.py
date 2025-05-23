import numpy
import matplotlib.cm
import centrosome.cpmorphology
DEFAULT_COLORMAP = "Default"

# TODO: Move appropriate functions to cellprofiler_library/functions

def get_default_colormap():
    return "jet"
def image_mode_black_and_white(pixel_data, mask, alpha, labels=None, colormap_value=None):
    pixel_data[mask] = True
    alpha[mask] = 1
    return pixel_data, alpha

def image_mode_grayscale(pixel_data, mask, alpha, labels, colormap_value=None):
    pixel_data[mask] = labels[mask].astype(float) / numpy.max(labels)
    alpha[mask] = 1
    return pixel_data, alpha

def image_mode_color(pixel_data, mask, alpha, labels, colormap_value):
    if colormap_value == DEFAULT_COLORMAP:
        cm_name = get_default_colormap()
    elif colormap_value == "colorcube":
        # Colorcube missing from matplotlib
        cm_name = "gist_rainbow"
    elif colormap_value == "lines":
        # Lines missing from matplotlib and not much like it,
        # Pretty boring palette anyway, hence
        cm_name = "Pastel1"
    elif colormap_value == "white":
        # White missing from matplotlib, it's just a colormap
        # of all completely white... not even different kinds of
        # white. And, isn't white just a uniform sampling of
        # frequencies from the spectrum?
        cm_name = "Spectral"
    else:
        cm_name = colormap_value

    cm = matplotlib.cm.get_cmap(cm_name)

    mapper = matplotlib.cm.ScalarMappable(cmap=cm)

    if labels.ndim == 3:
        for index, plane in enumerate(mask):
            pixel_data[index, plane, :] = mapper.to_rgba(
                centrosome.cpmorphology.distance_color_labels(labels[index])
            )[plane, :3]
    else:
        pixel_data[mask, :] += mapper.to_rgba(
            centrosome.cpmorphology.distance_color_labels(labels)
        )[mask, :3]

    alpha[mask] += 1
    return pixel_data, alpha

def image_mode_uint16(pixel_data, mask, alpha, labels, colormap_value=None):
    pixel_data[mask] = labels[mask]
    alpha[mask] = 1
    return pixel_data, alpha

def update_pixel_data(image_mode, objects_labels, objects_shape, colormap_value=None):
    
    alpha = numpy.zeros(objects_shape)

    fn_map = {
        "Binary (black & white)": image_mode_black_and_white,
        "Grayscale": image_mode_grayscale,
        "Color": image_mode_color,
        "uint16": image_mode_uint16,
    }

    pixel_data_init_map = {
        "Binary (black & white)": lambda: numpy.zeros(objects_shape, bool),
        "Grayscale": lambda: numpy.zeros(objects_shape),
        "Color": lambda: numpy.zeros(objects_shape + (3,)),
        "uint16": lambda: numpy.zeros(objects_shape, numpy.int32),
    }
    pixel_data = pixel_data_init_map.get(image_mode, lambda: numpy.zeros(objects_shape + (3,)))()
    for labels, _ in objects_labels:
        mask = labels != 0

        if numpy.all(~mask):
            continue
        pixel_data, alpha = fn_map[image_mode](pixel_data, mask, alpha, labels, colormap_value)
    mask = alpha > 0
    if image_mode == "Color":
        pixel_data[mask, :] = pixel_data[mask, :] / alpha[mask][:, numpy.newaxis]
    elif image_mode != "Binary (black & white)":
        pixel_data[mask] = pixel_data[mask] / alpha[mask]
    return pixel_data

