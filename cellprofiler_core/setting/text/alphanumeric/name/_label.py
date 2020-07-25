from ._name import Name
from ...._validation_error import ValidationError


class Label(Name):
    """
    A setting that provides an image name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(Label, self).__init__(text, "objectgroup", value, *args, **kwargs)

    def test_valid(self, pipeline):
        if self.value_text in ["Experiment", "Image", "Relationship"]:
            raise ValidationError(
                "Object names may not be any of %s"
                % (", ".join(["Experiment", "Image", "Relationship"])),
                self,
            )

        super(Label, self).test_valid(pipeline)
