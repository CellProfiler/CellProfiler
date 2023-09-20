class ChangeSettingEvent:
    """Abstract class representing either the event that a setting will be
    changed or has been changed

    """

    def __init__(self, old_value, new_value):
        self.__old_value = old_value
        self.__new_value = new_value

    def get_old_value(self):
        return self.__old_value

    old_value = property(get_old_value)

    def get_new_value(self):
        return self.__new_value

    new_value = property(get_new_value)
