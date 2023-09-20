from ._change_setting_event import ChangeSettingEvent


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
