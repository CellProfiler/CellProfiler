from ._name import Name
from ...._validation_error import ValidationError
from .....constants.pipeline import OBJECT_GROUP


class LabelName(Name):
    """
    A setting that provides an image name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(LabelName, self).__init__(text, OBJECT_GROUP, value, *args, **kwargs)

    def test_valid(self, pipeline):
        if self.value_text in ["Experiment", "Image", "Relationship"]:
            raise ValidationError(
                "Object names may not be any of %s"
                % (", ".join(["Experiment", "Image", "Relationship"])),
                self,
            )

        super(LabelName, self).test_valid(pipeline)
