from ..functions.image_processing import overlay_objects

def overlayobjects(
    image, labels, opacity=0.3, max_label=None, seed=None, colormap="jet"
):
    return overlay_objects(
        image=image, 
        labels=labels, 
        opacity=opacity, 
        max_label=max_label, 
        seed=seed,
        colormap=colormap
        )