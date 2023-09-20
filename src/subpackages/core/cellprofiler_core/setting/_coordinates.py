from ._setting import Setting
from ._validation_error import ValidationError


class Coordinates(Setting):
    """A setting representing X and Y coordinates on an image
    """

    def __init__(self, text, value=(0, 0), *args, **kwargs):
        """Initialize an integer range
        text  - helpful text to be displayed to the user
        value - initial default value, a two-tuple as x and y
        """
        super(Coordinates, self).__init__(text, "%d,%d" % value, *args, **kwargs)

    def set_value(self, value):
        """Convert integer tuples to string
        """
        try:
            if len(value) == 2:
                super(Coordinates, self).set_value("%d,%d" % (value[0], value[1]))
                return
        except:
            pass
        super(Coordinates, self).set_value(value)

    def get_value(self):
        """Convert the underlying string to a two-tuple"""
        return self.get_x(), self.get_y()

    def get_x_text(self):
        """Get the x coordinate as text"""
        return self.get_value_text().split(",")[0]

    def get_x(self):
        """The x coordinate"""
        return int(self.get_x_text())

    x = property(get_x)

    def get_y_text(self):
        vv = self.get_value_text().split(",")
        if len(vv) < 2:
            return ""
        return vv[1]

    def get_y(self):
        """The y coordinate"""
        return int(self.get_y_text())

    y = property(get_y)

    def test_valid(self, pipeline):
        values = self.value_text.split(",")
        if len(values) < 2:
            raise ValidationError("X and Y values must be separated by a comma", self)
        if len(values) > 2:
            raise ValidationError("Only two values allowed", self)
        for value in values:
            if not value.isdigit():
                raise ValidationError("%s is not an integer" % value, self)
