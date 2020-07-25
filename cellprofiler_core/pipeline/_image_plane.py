import json
import urllib.request

import javabridge


class ImagePlane:
    """This class represents the location and metadata for a 2-d image plane

    You need four pieces of information to reference an image plane:

    * The URL

    * The series number

    * The index

    * The channel # (to reference a monochrome plane in an interleaved image)

    In addition, image planes have associated metadata which is represented
    as a dictionary of keys and values.
    """

    MD_COLOR_FORMAT = "ColorFormat"
    MD_MONOCHROME = "monochrome"
    MD_RGB = "RGB"
    MD_PLANAR = "Planar"
    MD_SIZE_C = "SizeC"
    MD_SIZE_Z = "SizeZ"
    MD_SIZE_T = "SizeT"
    MD_SIZE_X = "SizeX"
    MD_SIZE_Y = "SizeY"
    MD_C = "C"
    MD_Z = "Z"
    MD_T = "T"
    MD_CHANNEL_NAME = "ChannelName"

    def __init__(self, jipd):
        self.jipd = jipd

    @property
    def path(self):
        """The file path if a file: URL, otherwise the URL"""
        if self.url.startswith("file:"):
            return urllib.request.url2pathname(self.url[5:])
        return self.url

    @property
    def url(self):
        return javabridge.run_script(
            "o.getImagePlane().getImageFile().getURI().toString()", dict(o=self.jipd)
        )

    @property
    def series(self):
        return javabridge.run_script(
            "o.getImagePlane().getSeries().getSeries()", dict(o=self.jipd)
        )

    @property
    def index(self):
        return javabridge.run_script("o.getImagePlane().getIndex()", dict(o=self.jipd))

    @property
    def channel(self):
        return javabridge.run_script(
            "o.getImagePlane().getChannel()", dict(o=self.jipd)
        )

    @property
    def metadata(self):
        return json.loads(
            javabridge.call(self.jipd, "jsonSerialize", "()Ljava/lang/String;")
        )
