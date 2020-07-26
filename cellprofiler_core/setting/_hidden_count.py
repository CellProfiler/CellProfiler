from ._setting import Setting


class HiddenCount(Setting):
    """A setting meant only for saving an item count

    The HiddenCount setting should never be in the visible settings.
    It should be tied to a sequence variable which gives the number of
    items which is the value for this variable.
    """

    def __init__(self, sequence, text="Hidden"):
        super(HiddenCount, self).__init__(text, str(len(sequence)))
        self.__sequence = sequence

    def set_value(self, value):
        if not value.isdigit():
            raise ValueError("The value must be an integer")
        count = int(value)
        if count == len(self.__sequence):
            # The value was "inadvertantly" set, but is correct
            return
        raise NotImplementedError(
            "The count should be inferred, not set  - actual: %d, set: %d"
            % (len(self.__sequence), count)
        )

    def get_value(self):
        return len(self.__sequence)

    def set_sequence(self, sequence):
        """Set the sequence used to maintain the count"""
        self.__sequence = sequence

    def __str__(self):
        return str(len(self.__sequence))

    def get_unicode_value(self):
        return str(len(self.__sequence))
