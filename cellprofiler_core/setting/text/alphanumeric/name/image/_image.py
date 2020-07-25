from cellprofiler_core.setting.text.alphanumeric.name._name import Name


class Image(Name):
    """A setting that provides an image name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(Image, self).__init__(text, "imagegroup", value, *args, **kwargs)
