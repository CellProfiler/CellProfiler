import os

import pkg_resources


def image_resource(filename):
    relpath = os.path.relpath(
        pkg_resources.resource_filename(
            "nucleus", os.path.join("data", "images", filename)
        )
    )

    # With this specific relative path we are probably building the documentation
    # in sphinx The path separator used by sphinx is "/" on all platforms.
    if relpath == os.path.join("..", "nucleus", "data", "images", filename):
        return "../images/{}".format(filename)

    # Otherwise, if you're rendering in the GUI, relative paths are fine
    # Note: the HTML renderer requires to paths to use '/' so we replace
    # the windows default '\\' here
    return relpath.replace("\\", "/")
