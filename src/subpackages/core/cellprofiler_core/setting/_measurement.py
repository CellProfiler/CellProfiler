from ._setting import Setting
from ._validation_error import ValidationError


class Measurement(Setting):
    """A measurement done on a class of objects (or Experiment or Image)

    A measurement represents a fully-qualified name of a measurement taken
    on an object. Measurements have categories and feature names and
    may or may not have a secondary image (for instance, the image used
    to measure an intensity), secondary object (for instance, the parent
    object when relating two classes of objects or the object name when
    aggregating object measurements over an image) or scale.
    """

    def __init__(self, text, object_fn, value="None", *args, **kwargs):
        """Construct the measurement category subscriber setting

        text - Explanatory text that appears to the side of the setting
        object_fn - a function that returns the measured object when called
        value - the initial value of the setting
        """
        super(Measurement, self).__init__(text, value, *args, **kwargs)
        self.__object_fn = object_fn

    @staticmethod
    def construct_value(category, feature_name, object_or_image_name, scale):
        """Construct a value that might represent a partially complete value"""
        if category is None:
            value = "None"
        elif feature_name is None:
            value = category
        else:
            parts = [category, feature_name]
            if object_or_image_name is not None:
                parts.append(object_or_image_name)
            if not scale is None:
                parts.append(scale)
            value = "_".join(parts)
        return str(value)

    def get_measurement_object(self):
        """Return the primary object for the measurement

        This is either "Image" if an image measurement or the name
        of the objects for per-object measurements. Please pardon the
        confusion with get_object_name which is the secondary object
        name, for instance for a measurement Relate."""
        return self.__object_fn()

    def get_category_choices(self, pipeline, object_name=None):
        """Find the categories of measurements available from the object """
        if object_name is None:
            object_name = self.__object_fn()
        categories = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            categories.update(module.get_categories(pipeline, object_name))
        result = list(categories)
        result.sort()
        return result

    def get_category(self, pipeline, object_name=None):
        """Return the currently chosen category"""
        categories = self.get_category_choices(pipeline, object_name)
        for category in categories:
            if self.value.startswith(category + "_") or self.value == category:
                return category
        return None

    def get_feature_name_choices(self, pipeline, object_name=None, category=None):
        """Find the feature name choices available for the chosen category"""
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline)
        if category is None:
            return []
        feature_names = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            feature_names.update(
                module.get_measurements(pipeline, object_name, category)
            )
        result = list(feature_names)
        result.sort()
        return result

    def get_feature_name(self, pipeline, object_name=None, category=None):
        """Return the currently selected feature name"""
        if category is None:
            category = self.get_category(pipeline, object_name)
        if category is None:
            return None
        feature_names = self.get_feature_name_choices(pipeline, object_name, category)
        for feature_name in feature_names:
            head = "_".join((category, feature_name))
            if self.value.startswith(head + "_") or self.value == head:
                return feature_name
        return None

    def get_image_name_choices(
        self, pipeline, object_name=None, category=None, feature_name=None
    ):
        """Find the secondary image name choices available for a feature

        A measurement can still be valid, even if there are no available
        image name choices. The UI should not offer image name choices
        if no choices are returned.
        """
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, category)
        if category is None or feature_name is None:
            return []
        image_names = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            image_names.update(
                module.get_measurement_images(
                    pipeline, object_name, category, feature_name
                )
            )
        result = list(image_names)
        result.sort()
        return result

    def get_image_name(
        self, pipeline, object_name=None, category=None, feature_name=None
    ):
        """Return the currently chosen image name"""
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if category is None:
            return None
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, category)
        if feature_name is None:
            return None
        image_names = self.get_image_name_choices(
            pipeline, object_name, category, feature_name
        )
        # 1st pass - accept only exact match
        # 2nd pass - accept part match.
        # This handles things like "OriginalBlue_Nuclei" vs "OriginalBlue"
        # in MeasureImageIntensity.
        #
        for full_match in True, False:
            for image_name in image_names:
                head = "_".join((category, feature_name, image_name))
                if (not full_match) and self.value.startswith(head + "_"):
                    return image_name
                if self.value == head:
                    return image_name
        return None

    def get_scale_choices(
        self,
        pipeline,
        object_name=None,
        category=None,
        feature_name=None,
        image_name=None,
    ):
        """Return the measured scales for the currently chosen measurement

        The setting may still be valid, even though there are no scale choices.
        In this case, the UI should not offer the user a scale choice.
        """
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, category)
        if image_name is None:
            image_name = self.get_image_name(
                pipeline, object_name, category, feature_name
            )
        if category is None or feature_name is None:
            return []
        scales = set()
        for module in pipeline.modules():
            if self.key() in [x.key() for x in module.settings()]:
                break
            scales.update(
                module.get_measurement_scales(
                    pipeline, object_name, category, feature_name, image_name
                )
            )
        result = [str(scale) for scale in scales]
        result.sort()
        return result

    def get_scale(
        self,
        pipeline,
        object_name=None,
        category=None,
        feature_name=None,
        image_name=None,
    ):
        """Return the currently chosen scale"""
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, category)
        if image_name is None:
            image_name = self.get_image_name(
                pipeline, object_name, category, feature_name
            )
        sub_object_name = self.get_object_name(pipeline)
        if category is None or feature_name is None:
            return None
        if image_name is not None:
            head = "_".join((category, feature_name, image_name))
        elif sub_object_name is not None:
            head = "_".join((category, feature_name, sub_object_name))
        else:
            head = "_".join((category, feature_name))
        for scale in self.get_scale_choices(pipeline):
            if self.value == "_".join((head, scale)):
                return scale
        return None

    def get_object_name_choices(
        self, pipeline, object_name=None, category=None, feature_name=None
    ):
        """Return a list of objects for a particular feature

        Typically these are image features measured on the objects in the image
        """
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, category)
        if any([x is None for x in (object_name, category, feature_name)]):
            return []
        objects = set()
        for module in pipeline.modules():
            if self.key in [x.key() for x in module.settings()]:
                break
            objects.update(
                module.get_measurement_objects(
                    pipeline, object_name, category, feature_name
                )
            )
        result = list(objects)
        result.sort()
        return result

    def get_object_name(
        self, pipeline, object_name=None, category=None, feature_name=None
    ):
        """Return the currently chosen image name"""
        if object_name is None:
            object_name = self.__object_fn()
        if category is None:
            category = self.get_category(pipeline, object_name)
        if category is None:
            return None
        if feature_name is None:
            feature_name = self.get_feature_name(pipeline, object_name, category)
        if feature_name is None:
            return None
        object_names = self.get_object_name_choices(
            pipeline, object_name, category, feature_name
        )
        for object_name in object_names:
            head = "_".join((category, feature_name, object_name))
            if self.value.startswith(head + "_") or self.value == head:
                return object_name
        return None

    def test_valid(self, pipeline):
        obname = self.__object_fn()
        category = self.get_category(pipeline, obname)
        if category is None:
            raise ValidationError(
                "%s has an unavailable measurement category" % self.value, self
            )
        feature_name = self.get_feature_name(pipeline, obname, category)
        if feature_name is None:
            raise ValidationError(
                "%s has an unmeasured feature name" % self.value, self
            )
        #
        # If there are any image names or object names, then there must
        # be a valid image name or object name
        #
        image_name = self.get_image_name(pipeline, obname, category, feature_name)
        image_names = self.get_image_name_choices(
            pipeline, obname, category, feature_name
        )
        sub_object_name = self.get_object_name(pipeline, obname, category, feature_name)
        sub_object_names = self.get_object_name_choices(
            pipeline, obname, category, feature_name
        )
        if len(sub_object_names) > 0 and image_name is None and sub_object_name is None:
            raise ValidationError(
                "%s has an unavailable object name" % self.value, self
            )
        if len(image_names) > 0 and image_name is None and sub_object_name is None:
            raise ValidationError("%s has an unavailable image name" % self.value, self)
        scale_choices = self.get_scale_choices(pipeline, obname, category, feature_name)
        if (
            self.get_scale(pipeline, obname, category, feature_name)
            not in scale_choices
            and len(scale_choices) > 0
        ):
            raise ValidationError("%s has an unavailable scale" % self.value, self)
        for module in pipeline.modules():
            if self.key() in [s.key() for s in module.visible_settings()]:
                break
        if not any(
            [
                column[0] == obname and column[1] == self.value
                for column in pipeline.get_measurement_columns(module)
            ]
        ):
            raise ValidationError(
                "%s is not measured for %s" % (self.value, obname), self
            )
