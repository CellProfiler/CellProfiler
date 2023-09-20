import os

from ._does_not_predicate import DoesNotPredicate
from ._does_predicate import DoesPredicate
from ._filter_predicate import FilterPredicate
from .._file_collection_display import FileCollectionDisplay
from ...utilities.image import is_image_extension

IS_TIF_PREDICATE = FilterPredicate(
    "istif",
    '"tif", "tiff", "ome.tif" or "ome.tiff"',
    lambda x: x.lower() in (".tif", ".tiff", ".ome.tif", ".ome.tiff"),
    [],
    doc="The extension is associated with TIFF image files",
)
IS_JPEG_PREDICATE = FilterPredicate(
    "isjpeg",
    '"jpg" or "jpeg"',
    lambda x: x.lower() in (".jpg", ".jpeg"),
    [],
    doc="The extension is associated with JPEG image files",
)
IS_PNG_PREDICATE = FilterPredicate(
    "ispng",
    '"png"',
    lambda x: x.lower() == ".png",
    [],
    doc="The extension is associated with PNG image files",
)
IS_IMAGE_PREDICATE = FilterPredicate(
    "isimage",
    "the extension of an image file",
    is_image_extension,
    [],
    "Is an extension commonly associated with image files",
)
IS_FLEX_PREDICATE = FilterPredicate(
    "isflex",
    '"flex"',
    lambda x: x.lower() == ".flex",
    [],
    doc="The extension is associated with .flex files",
)
IS_MOVIE_PREDICATE = FilterPredicate(
    "ismovie",
    '"mov" or "avi"',
    lambda x: x.lower() in (".mov", ".avi"),
    [],
    doc="The extension is associated with movie files",
)


class ExtensionPredicate(FilterPredicate):
    """A predicate that operates on file extensions"""

    def __init__(self):
        subpredicates = (
            IS_TIF_PREDICATE,
            IS_JPEG_PREDICATE,
            IS_PNG_PREDICATE,
            IS_IMAGE_PREDICATE,
            IS_FLEX_PREDICATE,
            IS_MOVIE_PREDICATE,
        )
        predicates = [
            DoesPredicate(subpredicates, "Is"),
            DoesNotPredicate(subpredicates, "Is not"),
        ]
        FilterPredicate.__init__(
            self,
            "extension",
            "Extension",
            self.fn_filter,
            predicates,
            doc="The rule applies to the file extension",
        )

    @staticmethod
    def fn_filter(node_type__modpath__module, *args):
        """The ExtensionPredicate filter function

        If the element is a file, try the different predicates on
        all possible extension parsings.
        """
        (node_type, modpath, module) = node_type__modpath__module
        if node_type == FileCollectionDisplay.NODE_DIRECTORY:
            return None
        elif isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            filename = modpath[-2]
        else:
            filename = modpath[-1]
        exts = ""
        while True:
            filename, ext = os.path.splitext(filename)
            if len(filename) == 0 or len(ext) == 0:
                return False
            exts = ext + exts
            if args[0](exts, *args[1:]):
                return True

    def test_valid(self, pipeline, *args):
        self((FileCollectionDisplay.NODE_FILE, ["/imaging", "test.tif"], None,), *args)
