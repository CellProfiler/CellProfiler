from ._text import Text


class Filename(Text):
    """A setting that displays a file name

    optional arguments -
       get_directory_fn is a function that gets the initial directory
           for the browse button
       set_directory_fn is a function that sets the directory after browsing
       browse_msg - message at top of file browser
       exts - a list of tuples where the first is the user-displayed text
       and the second is the file filter for an extension, like
       [("Pipeline (*.cp)","*.cp")]
       mode - Controls whether a file-open or file-save dialog is displayed
              when the user browses.
              FilenameText.MODE_OPEN - open a file that must exist
              FilenameText.MODE_APPEND - open a file for modification or
                    create a new file (using the Save dialog)
              FilenameText.MODE_OVERWRITE - create a new file and warn the
                    user if the file exists and will be overwritten.
    """

    MODE_OPEN = "Open"
    MODE_APPEND = "Append"
    MODE_OVERWRITE = "Overwrite"

    def __init__(self, text, value, *args, **kwargs):
        kwargs = kwargs.copy()
        self.get_directory_fn = kwargs.pop("get_directory_fn", None)
        self.set_directory_fn = kwargs.pop("set_directory_fn", None)
        self.browse_msg = kwargs.pop("browse_msg", "Choose a file")
        self.exts = kwargs.pop("exts", None)
        self.mode = kwargs.pop("mode", self.MODE_OPEN)
        super(Filename, self).__init__(text, value, *args, **kwargs)
        self.browsable = True

    def set_browsable(self, val):
        self.browsable = val
