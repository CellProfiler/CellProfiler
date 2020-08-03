class SettingEditedEvent:
    """Represents an attempt by the user to edit a setting

    """

    def __init__(self, setting, module, proposed_value, event):
        self.__module = module
        self.__setting = setting
        self.__proposed_value = proposed_value
        self.__event = event
        self.__accept_change = True

    def get_setting(self):
        """Return the setting being edited

        """
        return self.__setting

    def get_proposed_value(self):
        """Return the value proposed by the user

        """
        return self.__proposed_value

    def get_module(self):
        """Get the module holding the setting"""
        return self.__module

    def cancel(self):
        self.__accept_change = False

    def accept_change(self):
        return self.__accept_change

    def ui_event(self):
        """The event from the UI that triggered the edit

        """
        return self.__event
