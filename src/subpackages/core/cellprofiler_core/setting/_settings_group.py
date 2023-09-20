class SettingsGroup:
    """A group of settings that are managed together in the UI.
    Particulary useful when used with a RemoveSettingButton.
    Individual settings can be added with append(), and their value
    fetched from the group using the name given in append.
    """

    def __init__(self):
        self.settings = []

    def append(self, name, setting):
        """Add a new setting to the group, with a name.  The setting
        will then be available as group.name
        """
        assert name not in self.__dict__, (
            "%s already in SettingsGroup (previous setting or built in attribute)"
            % name
        )
        self.__setattr__(name, setting)
        self.settings.append(setting)

    def visible_settings(self):
        """Return a list of the settings in the group, in the order
        they were added to the group.
        """
        # return a copy
        return list(self.settings)

    def pipeline_settings(self):
        """Return a list of the settings, filtering out UI tidbits"""
        return [setting for setting in self.settings if setting.save_to_pipeline]
