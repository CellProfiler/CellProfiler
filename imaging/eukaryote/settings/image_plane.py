class ImagePlane(Setting):
    """A setting that specifies an image plane

    This setting lets the user pick an image plane. This might be an image
    file, a URL, or a plane from an image stack.

    The text setting has four fields, delimited by a space character (which
    is luckily disallowed in a URL:

    Field 1: URL

    Field 2: series (or None if a space is followed by another)

    Field 3: index (or None if blank)

    Field 4: channel (or None if blank)
    """

    def __init__(self, text, *args, **kwargs):
        '''Initialize the setting

        text - informative text to display to the left
        '''
        super(ImagePlane, self).__init__(
            text, ImagePlane.build(""), *args, **kwargs)

    @staticmethod
    def build(url, series=None, index=None, channel=None):
        '''Build the string representation of the setting

        url - the URL of the file containing the plane

        series - the series for a multi-series stack or None if the whole file

        index - the index of the frame for a multi-frame stack or None if
                the whole stack

        channel - the channel of an interlaced color image or None if all
                  channels
        '''
        if " " in url:
            # Spaces are not legal characters in URLs, nevertheless, I try
            # to accomodate
            logger.warn(
                "URLs should not contain spaces. %s is the offending URL" % url)
            url = url.replace(" ", "%20")
        return " ".join([str(x) if x is not None else ""
                         for x in url, series, index, channel])

    def __get_field(self, index):
        f = self.value_text.split(" ")[index]
        if len(f) == 0:
            return None
        return f

    def __get_int_field(self, index):
        f = self.__get_field(index)
        if f is None:
            return f
        return int(f)

    @property
    def url(self):
        '''The URL portion of the image plane descriptor'''
        uurl = self.__get_field(0)
        if uurl is not None:
            uurl = uurl.encode("utf-8")
        return uurl

    @property
    def series(self):
        '''The series portion of the image plane descriptor'''
        return self.__get_int_field(1)

    @property
    def index(self):
        '''The index portion of the image plane descriptor'''
        return self.__get_int_field(2)

    @property
    def channel(self):
        '''The channel portion of the image plane descriptor'''
        return self.__get_int_field(3)

    def test_valid(self, pipeline):
        if self.url is None:
            raise ValidationError(
                "This setting's URL is blank. Please select a valid image",
                self)
