from ._setting import Setting


class PathListDisplay(Setting):
    """This setting's only purpose is to signal that the path list should be shown

    Set self.using_filter to True if the module knows that the path list will
    be filtered or if the module doesn't know. Set it to False if the module
    knows the path list won't be filtered.
    """

    def __init__(self):
        super(self.__class__, self).__init__("", value="")
        self.using_filter = True
