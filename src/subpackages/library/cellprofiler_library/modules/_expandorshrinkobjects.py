from ..functions.object_processing import add_dividing_lines, despur, expand_defined_pixels, expand_until_touching, shrink_defined_pixels, shrink_to_point, skeletonize

def expand_or_shrink_objects(mode,labels,fill=None,iterations=None):
    if mode == 'expand_defined_pixels':
        return expand_defined_pixels(labels,iterations=iterations)
    elif mode == 'expand_infinite':
        return expand_until_touching(labels)
    elif mode == 'shrink_defined_pixels':
        return shrink_defined_pixels(labels,fill=fill,iterations=iterations)
    elif mode == 'shrink_to_point':
        return shrink_to_point(labels,fill=fill)
    elif mode == 'add_dividing_lines':
        return add_dividing_lines(labels)
    elif mode == 'despur':
        return despur(labels,iterations=iterations)
    elif mode == 'skeletonize':
        return skeletonize(labels)