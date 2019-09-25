from . import _setting


class DoSomething(_setting.Setting):
    """Do something in response to a button press
    """

    save_to_pipeline = False

    def __init__(self, text, label, callback, *args, **kwargs):
        super(DoSomething, self).__init__(text, "n/a", **kwargs)
        self.__label = label
        self.__callback = callback
        self.__args = args

    def get_label(self):
        """Return the text label for the button"""
        return self.__label

    def set_label(self, label):
        self.__label = label

    label = property(get_label, set_label)

    def on_event_fired(self):
        """Call the callback in response to the user's request to do something"""
        self.__callback(*self.__args)


class RemoveSettingButton(DoSomething):
    """A button whose only purpose is to remove something from a list."""

    def __init__(self, text, label, list, entry, **kwargs):
        super(RemoveSettingButton, self).__init__(
            text, label, lambda: list.remove(entry), **kwargs
        )


class PathListRefreshButton(DoSomething):
    """A setting that displays as a button which refreshes the path list"""

    def __init__(self, text, label, *args, **kwargs):
        DoSomething.__init__(self, text, label, self.fn_callback, *args, **kwargs)
        # callback set by module view
        self.callback = None

    def fn_callback(self, *args, **kwargs):
        if self.callback is not None:
            self.callback(*args, **kwargs)


class ImageSetDisplay(DoSomething):
    """A button that refreshes the image set display when pressed

    """

    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(
            args[0], args[1], None, *args[:2], **kwargs
        )
