import json

from ._setting import Setting


class DataTypes(Setting):
    """The DataTypes setting assigns data types to measurement names

    Imported or extracted metadata might be textual or numeric and
    that interpretation should be up to the user. This setting lets
    the user pick the data type for their metadata.
    """

    DT_TEXT = "text"
    DT_INTEGER = "integer"
    DT_FLOAT = "float"
    DT_NONE = "none"

    def __init__(self, text, value="{}", name_fn=None, *args, **kwargs):
        """Initializer

        text - description of the setting

        value - initial value (a json-encodable key/value dictionary)

        name_fn - a function that returns the current list of feature names
        """
        super(DataTypes, self).__init__(text, value, *args, **kwargs)

        self.__name_fn = name_fn

    def get_data_types(self):
        """Get a dictionary of the data type for every name

        Using the name function, if present, create a dictionary of name
        to data type (DT_TEXT / INTEGER / FLOAT / NONE)
        """
        result = json.loads(self.value_text)
        if self.__name_fn is not None:
            for name in self.__name_fn():
                if name not in result:
                    result[name] = self.DT_TEXT
        return result

    @staticmethod
    def decode_data_types(s):
        return json.loads(s)

    @staticmethod
    def encode_data_types(d):
        """Encode a data type dictionary as a potential value for this setting"""
        return json.dumps(d)
