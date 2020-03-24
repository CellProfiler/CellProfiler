class ImageSetChannelDescriptor:
    """This class represents the metadata for one image set channel

    An image set has a collection of channels which are either planar
    images or objects. The ImageSetChannelDescriptor describes one
    of these:

    The channel's name

    The channel's type - grayscale image / color image / objects / mask
    or illumination function
    """

    # Channel types
    CT_GRAYSCALE = "Grayscale"
    CT_COLOR = "Color"
    CT_MASK = "Mask"
    CT_OBJECTS = "Objects"
    CT_FUNCTION = "Function"

    def __init__(self, name, channel_type):
        self.name = name
        self.channel_type = channel_type
