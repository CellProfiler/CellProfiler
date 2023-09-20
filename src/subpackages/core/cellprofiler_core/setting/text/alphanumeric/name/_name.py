from .._alphanumeric import Alphanumeric


class Name(Alphanumeric):
    """
    A setting that provides a named object
    """

    def __init__(self, text, group, value="Do not use", *args, **kwargs):
        self.__provided_attributes = {"group": group}
        kwargs = kwargs.copy()
        if "provided_attributes" in kwargs:
            self.__provided_attributes.update(kwargs["provided_attributes"])
            del kwargs["provided_attributes"]
        kwargs["first_must_be_alpha"] = True
        super(Name, self).__init__(text, value, *args, **kwargs)

    def get_group(self):
        """
        This setting provides a name to this group

        Returns a group name, e.g., imagegroup or objectgroup
        """
        return self.__provided_attributes["group"]

    group = property(get_group)

    @property
    def provided_attributes(self):
        """
        Return the dictionary of attributes of this provider

        These are things like the group ("objectgroup" for instance) and
        hints about the thing itself, such as that it is an image
        that was loaded from  a file.
        """
        return self.__provided_attributes
