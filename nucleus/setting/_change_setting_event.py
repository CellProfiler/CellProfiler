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


class BeforeChangeSettingEvent(ChangeSettingEvent):
    """Indicates that a setting is about to change, allows a listener to cancel the change

    """

    def __init__(self, old_value, new_value):
        ChangeSettingEvent.__init__(self, old_value, new_value)
        self.__allow_change = True
        self.__cancel_reason = None

    def cancel_change(self, reason=None):
        self.__allow_change = False
        self.__cancel_reason = reason

    def allow_change(self):
        return self.__allow_change

    def get_cancel_reason(self):
        return self.__cancel_reason

    cancel_reason = property(get_cancel_reason)


class AfterChangeSettingEvent(ChangeSettingEvent):
    """Indicates that a setting has changed its value

    """

    def __init__(self, old_value, new_value):
        ChangeSettingEvent.__init__(self, old_value, new_value)
