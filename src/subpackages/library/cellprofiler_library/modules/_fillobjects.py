from ..functions.object_processing import fill_object_holes, fill_convex_hulls

def fillobjects(labels, mode="holes", diameter=64.0, planewise=False):
    if mode.casefold() == "holes":
        return fill_object_holes(labels, diameter, planewise)
    elif mode.casefold() in ("convex hull", "convex_hull"):
        return fill_convex_hulls(labels)
    else:
        raise ValueError(f"Mode '{mode}' is not supported. Available modes are: 'holes' and 'convex_hull'.")

