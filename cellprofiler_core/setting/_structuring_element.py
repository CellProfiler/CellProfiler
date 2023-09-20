import skimage.morphology

from ._setting import Setting
from ._validation_error import ValidationError


class StructuringElement(Setting):
    def __init__(
        self,
        text="Structuring element",
        value="Disk,1",
        allow_planewise=False,
        *args,
        **kwargs,
    ):
        self.__allow_planewise = allow_planewise

        super(StructuringElement, self).__init__(text, value, *args, **kwargs)

    @staticmethod
    def get_choices():
        return ["Ball", "Cube", "Diamond", "Disk", "Octahedron", "Square", "Star"]

    def get_value(self):
        return getattr(skimage.morphology, self.shape)(self.size)

    def set_value(self, value):
        self.value_text = value

    @property
    def shape(self):
        return str(self.value_text.split(",")[0]).lower()

    @shape.setter
    def shape(self, value):
        self.value_text = ",".join((value, str(self.size)))

    @property
    def size(self):
        _, size = self.value_text.split(",")

        return int(size) if size else None

    @size.setter
    def size(self, value):
        self.value_text = ",".join((self.shape, str(value)))

    def test_valid(self, pipeline):
        if self.size is None:
            raise ValidationError(
                "Missing structuring element size. Please enter a positive integer.",
                self,
            )

        if self.size <= 0:
            raise ValidationError(
                "Structuring element size must be a positive integer. You provided {}.".format(
                    self.size
                ),
                self,
            )

        if pipeline.volumetric():
            if (
                self.shape in ["diamond", "disk", "square", "star"]
                and not self.__allow_planewise
            ):
                raise ValidationError(
                    "A 3 dimensional struturing element is required. You selected {}."
                    ' Please select one of "ball", "cube", or "octahedron".'.format(
                        self.shape
                    ),
                    self,
                )
        else:
            if self.shape in ["ball", "cube", "octahedron"]:
                raise ValidationError(
                    "A 2 dimensional structuring element is required. You selected {}."
                    ' Please select one of "diamond", "disk", "square", "star".'.format(
                        self.shape
                    ),
                    self,
                )
