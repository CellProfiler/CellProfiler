class ValidationError(ValueError):
    """An exception indicating that a setting's value prevents the pipeline from running
    """

    def __init__(self, message, setting):
        """Initialize with an explanatory message and the setting that caused the problem
        """
        super(ValidationError, self).__init__(message)
        self.message = message
        self.__setting = setting

    def get_setting(self):
        """The setting responsible for the problem

        This might be one of several settings partially responsible
        for the problem.
        """
        return self.__setting

    setting = property(get_setting)
