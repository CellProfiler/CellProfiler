from ._setting import Setting


class HTMLText(Setting):
    """The HTMLText setting displays a HTML control with content

    """

    def __init__(self, text, content="", size=None, **kwargs):
        """Initialize with the html content

        text - the text to the right of the setting

        content - the HTML to display

        size - a (x,y) tuple of the minimum window size in units of
               wx.SYS_CAPTION_Y (the height of the window caption).
        """
        super(self.__class__, self).__init__(text, "", **kwargs)
        self.content = content
        self.size = size
