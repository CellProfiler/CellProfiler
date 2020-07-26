from ._change_setting_event import ChangeSettingEvent


class AfterChangeSettingEvent(ChangeSettingEvent):
    """Indicates that a setting has changed its value

    """

    def __init__(self, old_value, new_value):
        ChangeSettingEvent.__init__(self, old_value, new_value)
