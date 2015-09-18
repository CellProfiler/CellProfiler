class Number(Text):
    """A setting that allows only numeric input
    """

    def __init__(self, text, value=0, minval=None, maxval=None, *args,
                 **kwargs):
        if isinstance(value, basestring):
            text_value = value
            value = self.str_to_value(value)
        else:
            text_value = self.value_to_str(value)
        super(Number, self).__init__(text, text_value, *args, **kwargs)
        self.__default = self.str_to_value(text_value)
        self.__minval = minval
        self.__maxval = maxval

    def str_to_value(self, str_value):
        """Return the value of the string passed

        Override this in a derived class to parse the numeric text or
        raise an exception if badly formatted.
        """
        raise NotImplementedError("Please define str_to_value in a subclass")

    def value_to_str(self, value):
        """Return the string representation of the value passed

        Override this in a derived class to convert a numeric value into text
        """
        raise NotImplementedError("Please define value_to_str in a subclass")

    def set_value(self, value):
        """Convert integer to string
        """
        str_value = unicode(value) if isinstance(value, basestring) \
            else self.value_to_str(value)
        self.set_value_text(str_value)

    def get_value(self, reraise=False):
        """Return the value of the setting as a float
        """
        return self.__default

    def set_value_text(self, value_text):
        super(Number, self).set_value_text(value_text)
        try:
            self.test_valid(None)
            self.__default = self.str_to_value(value_text)
        except:
            logger.debug("Number set to illegal value: %s" % value_text)

    def test_valid(self, pipeline):
        """Return true only if the text value is float
        """
        try:
            value = self.str_to_value(self.value_text)
        except ValueError:
            raise ValidationError('Value not in decimal format', self)
        if self.__minval is not None and self.__minval > value:
            raise ValidationError(
                'Must be at least %s, was %s' %
                (self.value_to_str(self.__minval), self.value_text), self)
        if self.__maxval is not None and self.__maxval < value:
            raise ValidationError(
                'Must be at most %s, was %s' %
                (self.value_to_str(self.__maxval), self.value_text), self)

    def eq(self, x):
        '''Equal if our value equals the operand'''
        return self.value == x
