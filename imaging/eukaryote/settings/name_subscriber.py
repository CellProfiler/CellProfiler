class NameSubscriber(Setting):
    """A setting that takes its value from one made available by name providers
    """

    def __init__(self, text, group, value=None,
                 can_be_blank=False, blank_text=LEAVE_BLANK, *args, **kwargs):
        if value is None:
            value = (can_be_blank and blank_text) or "None"
        self.__required_attributes = {"group": group}
        if kwargs.has_key(REQUIRED_ATTRIBUTES):
            self.__required_attributes.update(kwargs[REQUIRED_ATTRIBUTES])
            kwargs = kwargs.copy()
            del kwargs[REQUIRED_ATTRIBUTES]
        self.__can_be_blank = can_be_blank
        self.__blank_text = blank_text
        super(NameSubscriber, self).__init__(text, value, *args, **kwargs)

    def get_group(self):
        """This setting provides a name to this group

        Returns a group name, e.g. imagegroup or objectgroup
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
        return all([setting.provided_attributes.get(key, None) ==
                    self.__required_attributes[key]
                    for key in self.__required_attributes.keys()])

    def test_valid(self, pipeline):
        choices = self.get_choices(pipeline)
        if len(choices) == 0:
            raise ValidationError("No prior instances of %s were defined" % (self.group), self)
        if self.value not in [c[0] for c in choices]:
            raise ValidationError("%s not in %s" % (self.value, ", ".join(c[0] for c in self.get_choices(pipeline))),
                                  self)
