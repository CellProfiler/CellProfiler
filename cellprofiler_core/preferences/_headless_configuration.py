import json


class HeadlessConfiguration:
    """
    This class functions as a configuration set for headless runs.
    The default config class is wx-based, which means we have to replace it
    here with this psuedo-replacement that has the same interface
    """

    def __init__(self):
        self.__preferences = {}

    def Read(self, kwd):
        return self.__preferences[kwd]

    def ReadInt(self, kwd, default=0):
        return int(self.__preferences.get(kwd, default))

    def ReadBool(self, kwd, default=False):
        return bool(self.__preferences.get(kwd, default))

    def Write(self, kwd, value):
        self.__preferences[kwd] = value

    # wx implements these for their own version of the "Config" object
    # Because this class is a mock config object without wx, we need to
    # make its interface the same
    WriteInt = Write
    WriteBool = Write

    def Exists(self, kwd):
        return kwd in self.__preferences

    def GetEntryType(self, kwd):
        """Get the data type of the registry key.

        Returns wx.Config.Type_String = 1
        """
        return 1

    def load_json(self, path):
        with open(path) as fd:
            config_dict = json.load(fd)
        self.__preferences.update(config_dict)
