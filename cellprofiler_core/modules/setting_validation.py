import os.path
import urllib.parse

from ..module import Module
from ..pipeline import ImagePlane as ImagePlane
from ..utilities.pathname import pathname2url


class SettingValidation(Module):
    """A fake module for setting validation"""

    @staticmethod
    def get_image_plane_details(modpath):
        if modpath[0] in ("http", "https", "ftp", "s3", "gs"):
            if len(modpath) == 1:
                return modpath[0] + ":"
            elif len(modpath) == 2:
                return modpath[0] + ":" + modpath[1]
            else:
                return (
                    modpath[0]
                    + ":"
                    + modpath[1]
                    + "/"
                    + "/".join([urllib.parse.quote(part) for part in modpath[2:]])
                )

        path = os.path.join(*modpath)

        url = pathname2url(path)

        return ImagePlane(ImageFile(url))
