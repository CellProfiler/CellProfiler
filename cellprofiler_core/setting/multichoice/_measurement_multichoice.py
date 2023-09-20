from ._multichoice import MultiChoice


class MeasurementMultiChoice(MultiChoice):
    """A multi-choice setting for selecting multiple measurements"""

    def __init__(self, text, value="", *args, **kwargs):
        """Initialize the measurement multi-choice

        At initialization, the choices are empty because the measurements
        can't be fetched here. It's done (bit of a hack) in test_valid.
        """
        super(MeasurementMultiChoice, self).__init__(text, [], value, *args, **kwargs)

    @staticmethod
    def encode_object_name(object_name):
        """Encode object name, escaping |"""
        return object_name.replace("|", "||")

    @staticmethod
    def decode_object_name(object_name):
        """Decode the escaped object name"""
        return object_name.replace("||", "|")

    @staticmethod
    def split_choice(choice):
        """Split object and feature within a choice"""
        subst_choice = choice.replace("||", "++")
        loc = subst_choice.find("|")
        if loc == -1:
            return subst_choice, "Invalid"
        return choice[:loc], choice[(loc + 1) :]

    def get_measurement_object(self, choice):
        return self.decode_object_name(self.split_choice(choice)[0])

    def get_measurement_feature(self, choice):
        return self.split_choice(choice)[1]

    def make_measurement_choice(self, object_name, feature):
        return self.encode_object_name(object_name) + "|" + feature

    @staticmethod
    def get_value_string(choices):
        """Return the string value representing the choices made

        choices - a collection of choices as returned by make_measurement_choice
        """
        return ",".join(choices)

    def test_valid(self, pipeline):
        """Get the choices here and call the superclass validator"""
        self.populate_choices(pipeline)
        super(MeasurementMultiChoice, self).test_valid(pipeline)

    def populate_choices(self, pipeline):
        #
        # Find our module
        #
        for module in pipeline.modules():
            for setting in module.visible_settings():
                if id(setting) == id(self):
                    break
        columns = pipeline.get_measurement_columns(module)

        def valid_mc(c):
            """Disallow any measurement column with "," or "|" in its names"""
            return not any([any([bad in f for f in c[:2]]) for bad in (",", "|")])

        self.set_choices(
            [self.make_measurement_choice(c[0], c[1]) for c in columns if valid_mc(c)]
        )
