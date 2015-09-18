class ObjectNameProvider(NameProvider):
    """A setting that provides an image name
    """

    def __init__(self, text, value=DO_NOT_USE, *args, **kwargs):
        super(ObjectNameProvider, self).__init__(text, OBJECT_GROUP, value,
                                                 *args, **kwargs)

    def test_valid(self, pipeline):
        if self.value_text in cellprofiler.measurements.disallowed_object_names:
            raise ValidationError(
                "Object names may not be any of %s" % (", ".join(cellprofiler.measurements.disallowed_object_names)),
                self)
        super(ObjectNameProvider, self).test_valid(pipeline)
