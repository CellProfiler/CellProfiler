from ._filter_predicate import FilterPredicate
from ._does_predicate import DoesPredicate
from ._does_not_predicate import DoesNotPredicate
from .._file_collection_display import FileCollectionDisplay
from ...constants.image import MD_SIZE_T, MD_SIZE_Z
from ...constants.measurement import C_RGB, C_MONOCHROME


class ImagePredicate(FilterPredicate):
    """A predicate that applies subpredicates to image plane details"""

    IS_COLOR_PREDICATE = FilterPredicate(
        "iscolor",
        "Color",
        lambda plane: (
                plane.color_format == C_RGB
        ),
        [],
        doc="The image is an interleaved color image (for example, a PNG image)",
    )

    IS_MONOCHROME_PREDICATE = FilterPredicate(
        "ismonochrome",
        "Monochrome",
        lambda plane: (
            plane.color_format == C_MONOCHROME
        ),
        [],
        doc="The image is monochrome",
    )

    @staticmethod
    def is_stack(x):
        if MD_SIZE_T in x.metadata and x.metadata[MD_SIZE_T] > 1:
            return True
        if MD_SIZE_Z in x.metadata and x.metadata[MD_SIZE_Z] > 1:
            return True
        return False

    IS_STACK_PREDICATE = FilterPredicate(
        "isstack",
        "Stack",
        lambda x: ImagePredicate.is_stack(x),
        [],
        doc="The image is a Z-stack or movie",
    )

    IS_STACK_FRAME_PREDICATE = FilterPredicate(
        "isstackframe",
        "Stack frame",
        lambda x: x.index is not None,
        [],
        doc="The image is a frame of a movie or a plane of a Z-stack",
    )

    def __init__(self):
        subpredicates = (
            self.IS_COLOR_PREDICATE,
            self.IS_MONOCHROME_PREDICATE,
            self.IS_STACK_PREDICATE,
            self.IS_STACK_FRAME_PREDICATE,
        )
        predicates = [
            pred_class(subpredicates, text)
            for pred_class, text in (
                (DoesPredicate, "Is"),
                (DoesNotPredicate, "Is not"),
            )
        ]

        FilterPredicate.__init__(
            self,
            "image",
            "Image",
            self.fn_filter,
            predicates,
            doc="Filter based on image characteristics",
        )

    @staticmethod
    def fn_filter(node_type__modpath__plane, *args):
        (node_type, modpath, plane) = node_type__modpath__plane
        if node_type == FileCollectionDisplay.NODE_DIRECTORY:
            return None
        if plane is None:
            return None
        return args[0](plane, *args[1:])

    def test_valid(self, pipeline, *args):
        from ...modules.setting_validation import SettingValidation

        self(
            (
                FileCollectionDisplay.NODE_FILE,
                ["/imaging", "test.tif"],
                SettingValidation(),
            ),
            *args,
        )
