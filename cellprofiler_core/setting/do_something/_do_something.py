from .._setting import Setting


class DoSomething(Setting):
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
