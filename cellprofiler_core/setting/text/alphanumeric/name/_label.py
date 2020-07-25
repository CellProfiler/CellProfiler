import cellprofiler_core.measurement
from cellprofiler_core.setting import Name, ValidationError


class Label(Name):
    """A setting that provides an image name
    """

    def __init__(self, text, value="Do not use", *args, **kwargs):
        super(Label, self).__init__(
            text, "objectgroup", value, *args, **kwargs
        )

    def test_valid(self, pipeline):
        if self.value_text in cellprofiler_core.measurement.disallowed_object_names:
            raise ValidationError(
                "Object names may not be any of %s"
                % (", ".join(cellprofiler_core.measurement.disallowed_object_names)),
                self,
            )
        super(Label, self).test_valid(pipeline)
