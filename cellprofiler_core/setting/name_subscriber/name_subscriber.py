from cellprofiler_core.setting._setting import Setting
from cellprofiler_core.setting._validation_error import ValidationError
from cellprofiler_core.setting.text import (
    get_name_provider_choices,
    OutlineNameProvider,
)


class NameSubscriber(Setting):
    """A setting that takes its value from one made available by name providers
    """

    def __init__(
        self,
        text,
        group,
        value=None,
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        if value is None:
            value = (can_be_blank and blank_text) or "None"
        self.__required_attributes = {"group": group}
        if "required_attributes" in kwargs:
            self.__required_attributes.update(kwargs["required_attributes"])
            kwargs = kwargs.copy()
            del kwargs["required_attributes"]
        self.__can_be_blank = can_be_blank
        self.__blank_text = blank_text
        super(NameSubscriber, self).__init__(text, value, *args, **kwargs)

    def get_group(self):
        """This setting provides a name to this group

        Returns a group name, e.g., imagegroup or objectgroup
        """
        return self.__required_attributes["group"]

    group = property(get_group)

    def get_choices(self, pipeline):
        choices = []
        if self.__can_be_blank:
            choices.append((self.__blank_text, "", 0, False))
        return choices + sorted(get_name_provider_choices(pipeline, self, self.group))

    def get_is_blank(self):
        """True if the selected choice is the blank one"""
        return self.__can_be_blank and self.value == self.__blank_text

    is_blank = property(get_is_blank)

    def matches(self, setting):
        """Return true if this subscriber matches the category of the provider"""
        return all(
            [
                setting.provided_attributes.get(key, None)
                == self.__required_attributes[key]
                for key in list(self.__required_attributes.keys())
            ]
        )

    def test_valid(self, pipeline):
        choices = self.get_choices(pipeline)
        if len(choices) == 0:
            raise ValidationError(
                "No prior instances of %s were defined" % self.group, self
            )
        if self.value not in [c[0] for c in choices]:
            raise ValidationError(
                "%s not in %s"
                % (self.value, ", ".join(c[0] for c in self.get_choices(pipeline))),
                self,
            )


class ImageNameSubscriber(NameSubscriber):
    """A setting that provides an image name
    """

    def __init__(
        self,
        text,
        value=None,
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(ImageNameSubscriber, self).__init__(
            text, "imagegroup", value, can_be_blank, blank_text, *args, **kwargs
        )


class FileImageNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names loaded from files"""

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        kwargs = kwargs.copy()
        if "required_attributes" not in kwargs:
            kwargs["required_attributes"] = {}
        kwargs["required_attributes"]["file_image"] = True
        super(FileImageNameSubscriber, self).__init__(
            text, value, can_be_blank, blank_text, *args, **kwargs
        )


class CroppingNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names that have cropping masks"""

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        kwargs = kwargs.copy()
        if "required_attributes" not in kwargs:
            kwargs["required_attributes"] = {}
        kwargs["required_attributes"]["cropping_image"] = True
        super(CroppingNameSubscriber, self).__init__(
            text, value, can_be_blank, blank_text, *args, **kwargs
        )


class ExternalImageNameSubscriber(ImageNameSubscriber):
    """A setting that provides image names loaded externally (eg: from Java)"""

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(ExternalImageNameSubscriber, self).__init__(
            text, value, can_be_blank, blank_text, *args, **kwargs
        )


class ObjectNameSubscriber(NameSubscriber):
    """A setting that subscribes to the list of available object names
    """

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(ObjectNameSubscriber, self).__init__(
            text, "objectgroup", value, can_be_blank, blank_text, *args, **kwargs
        )


class OutlineNameSubscriber(ImageNameSubscriber):
    """A setting that subscribes to the list of available object outline names
    """

    def __init__(
        self,
        text,
        value="None",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(OutlineNameSubscriber, self).__init__(
            text, value, can_be_blank, blank_text, *args, **kwargs
        )

    def matches(self, setting):
        """Only match OutlineNameProvider variables"""
        return isinstance(setting, OutlineNameProvider)


class GridNameSubscriber(NameSubscriber):
    """A setting that subscribes to grid information providers
    """

    def __init__(
        self,
        text,
        value="Do not use",
        can_be_blank=False,
        blank_text="Leave blank",
        *args,
        **kwargs,
    ):
        super(GridNameSubscriber, self).__init__(
            text, "gridgroup", value, can_be_blank, blank_text, *args, **kwargs
        )
