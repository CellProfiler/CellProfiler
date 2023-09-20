from ._setting import Setting


class DoThings(Setting):
    """Do one of several things, depending on which button is pressed

    This setting consolidates several possible actions into one setting.
    Graphically, it displays as several buttons that are horizontally
    adjacent.
    """

    save_to_pipeline = False

    def __init__(self, text, labels_and_callbacks, *args, **kwargs):
        """Initializer

        text - text to display to left of setting

        labels_and_callbacks - a sequence of two tuples of button label
        and callback to be called

        All additional function arguments are passed to the callback.
        """
        super(DoThings, self).__init__(text, "n/a", **kwargs)
        self.__args = tuple(args)
        self.__labels_and_callbacks = labels_and_callbacks

    @property
    def count(self):
        """The number of things to do

        returns the number of buttons to display = number of actions
        that can be performed.
        """
        return len(self.__labels_and_callbacks)

    def get_label(self, idx):
        """Retrieve one of the actions' labels

        idx - the index of the action
        """
        return self.__labels_and_callbacks[idx][0]

    def set_label(self, idx, label):
        """Set the label for an action

        idx - the index of the action

        label - the label to display for that action
        """
        self.__labels_and_callbacks[idx] = (label, self.__labels_and_callbacks[idx][1])

    def on_event_fired(self, idx):
        """Call the indexed action's callback

        idx - index of the action to fire
        """
        self.__labels_and_callbacks[idx][1](*self.__args)
