import os

import cellprofiler.module
import cellprofiler.pipeline
import cellprofiler.setting
import cellprofiler.utilities.url
import javabridge


class DirectoryPredicate(cellprofiler.setting.Filter.FilterPredicate):
    """A predicate that only filters directories"""

    def __init__(self):
        subpredicates = (
            cellprofiler.setting.Filter.CONTAINS_PREDICATE,
            cellprofiler.setting.Filter.CONTAINS_REGEXP_PREDICATE,
            cellprofiler.setting.Filter.STARTS_WITH_PREDICATE,
            cellprofiler.setting.Filter.ENDSWITH_PREDICATE,
            cellprofiler.setting.Filter.EQ_PREDICATE)
        predicates = [cellprofiler.setting.Filter.DoesPredicate(subpredicates),
                      cellprofiler.setting.Filter.DoesNotPredicate(subpredicates)]
        cellprofiler.setting.Filter.FilterPredicate.__init__(self,
                                            'directory', "Directory", self.fn_filter,
                                                             predicates, doc="Apply the rule to directories")

    def fn_filter(self, (node_type, modpath, module), *args):
        """The DirectoryPredicate filter function

        The arg slot expects a tuple of node_type and modpath.
        The predicate returns None (= agnostic about filtering) if
        the node is not a directory, otherwise it composites the
        modpath into a file path and applies it to the rest of
        the args.
        """
        if isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            path = os.path.join(*modpath[:-2])
        else:
            path = os.path.join(*modpath[:-1])
        return args[0](path, *args[1:])

    def test_valid(self, pipeline, *args):
        self((cellprofiler.setting.FileCollectionDisplay.NODE_FILE,
              ["/imaging", "image.tif"], None), *args)


class FilePredicate(cellprofiler.setting.Filter.FilterPredicate):
    """A predicate that only filters files"""

    def __init__(self):
        subpredicates = (
            cellprofiler.setting.Filter.CONTAINS_PREDICATE,
            cellprofiler.setting.Filter.CONTAINS_REGEXP_PREDICATE,
            cellprofiler.setting.Filter.STARTS_WITH_PREDICATE,
            cellprofiler.setting.Filter.ENDSWITH_PREDICATE,
            cellprofiler.setting.Filter.EQ_PREDICATE)
        predicates = [cellprofiler.setting.Filter.DoesPredicate(subpredicates),
                      cellprofiler.setting.Filter.DoesNotPredicate(subpredicates)]
        cellprofiler.setting.Filter.FilterPredicate.__init__(self,
                                            'file', "File", self.fn_filter, predicates,
                                                             doc="Apply the rule to files")

    def fn_filter(self, (node_type, modpath, module), *args):
        """The FilePredicate filter function

        The arg slot expects a tuple of node_type and modpath.
        The predicate returns None (= agnostic about filtering) if
        the node is not a directory, otherwise it composites the
        modpath into a file path and applies it to the rest of
        the args
        """
        if node_type == cellprofiler.setting.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        elif isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            filename = modpath[-2]
        else:
            filename = modpath[-1]
        return args[0](filename, *args[1:])

    def test_valid(self, pipeline, *args):
        self((cellprofiler.setting.FileCollectionDisplay.NODE_FILE,
              ["/imaging", "test.tif"], None), *args)


class ExtensionPredicate(cellprofiler.setting.Filter.FilterPredicate):
    """A predicate that operates on file extensions"""
    IS_TIF_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            "istif", '"tif", "tiff", "ome.tif" or "ome.tiff"',
            lambda x: x.lower() in ("tif", "tiff", "ome.tif", "ome.tiff"), [],
            doc="The extension is associated with TIFF image files")
    IS_JPEG_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            "isjpeg", '"jpg" or "jpeg"',
            lambda x: x.lower() in ("jpg", "jpeg"), [],
            doc="The extension is associated with JPEG image files")
    IS_PNG_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            "ispng", '"png"',
            lambda x: x.lower() == "png", [],
            doc="The extension is associated with PNG image files")
    IS_IMAGE_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            'isimage', 'the extension of an image file',
            is_image_extension, [],
            'Is an extension commonly associated with image files')
    IS_FLEX_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            'isflex', '"flex"',
            lambda x: x.lower() == "flex", [],
            doc="The extension is associated with .flex files")
    IS_MOVIE_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            "ismovie", '"mov" or "avi"',
            lambda x: x.lower() in ("mov", "avi"), [],
            doc="The extension is associated with movie files")

    def __init__(self):
        subpredicates = (
            self.IS_TIF_PREDICATE,
            self.IS_JPEG_PREDICATE,
            self.IS_PNG_PREDICATE,
            self.IS_IMAGE_PREDICATE,
            self.IS_FLEX_PREDICATE,
            self.IS_MOVIE_PREDICATE)
        predicates = [cellprofiler.setting.Filter.DoesPredicate(subpredicates, "Is"),
                      cellprofiler.setting.Filter.DoesNotPredicate(subpredicates, "Is not")]
        cellprofiler.setting.Filter.FilterPredicate.__init__(self,
                                            'extension', "Extension", self.fn_filter, predicates,
                                                             doc="The rule applies to the file extension")

    def fn_filter(self, (node_type, modpath, module), *args):
        """The ExtensionPredicate filter function

        If the element is a file, try the different predicates on
        all possible extension parsings.
        """
        if node_type == cellprofiler.setting.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        elif isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            filename = modpath[-2]
        else:
            filename = modpath[-1]
        exts = []
        while True:
            filename, ext = os.path.splitext(filename)
            if len(filename) == 0 or len(ext) == 0:
                return False
            exts.insert(0, ext[1:])
            ext = '.'.join(exts)
            if args[0](ext, *args[1:]):
                return True

    def test_valid(self, pipeline, *args):
        self((cellprofiler.setting.FileCollectionDisplay.NODE_FILE,
              ["/imaging", "test.tif"], None), *args)


class ImagePredicate(cellprofiler.setting.Filter.FilterPredicate):
    """A predicate that applies subpredicates to image plane details"""
    IS_COLOR_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            "iscolor", "Color",
            lambda x: (
                x.metadata.has_key(cellprofiler.pipeline.ImagePlaneDetails.MD_COLOR_FORMAT) and
                x.metadata[cellprofiler.pipeline.ImagePlaneDetails.MD_COLOR_FORMAT] ==
                cellprofiler.pipeline.ImagePlaneDetails.MD_RGB), [],
            doc="The image is an interleaved color image (for example, a PNG image)")

    IS_MONOCHROME_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            "ismonochrome", "Monochrome",
            lambda x: (
                x.metadata.has_key(cellprofiler.pipeline.ImagePlaneDetails.MD_COLOR_FORMAT) and
                x.metadata[cellprofiler.pipeline.ImagePlaneDetails.MD_COLOR_FORMAT] ==
                cellprofiler.pipeline.ImagePlaneDetails.MD_MONOCHROME), [],
            doc="The image is monochrome")

    @staticmethod
    def is_stack(x):
        if (x.metadata.has_key(cellprofiler.pipeline.ImagePlaneDetails.MD_SIZE_T) and
                    x.metadata[cellprofiler.pipeline.ImagePlaneDetails.MD_SIZE_T] > 1):
            return True
        if (x.metadata.has_key(cellprofiler.pipeline.ImagePlaneDetails.MD_SIZE_Z) and
                    x.metadata[cellprofiler.pipeline.ImagePlaneDetails.MD_SIZE_Z] > 1):
            return True
        return False

    IS_STACK_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            "isstack", "Stack", lambda x: ImagePredicate.is_stack(x), [],
            doc="The image is a Z-stack or movie")

    IS_STACK_FRAME_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
            "isstackframe", "Stack frame", lambda x: x.index is not None, [],
            doc="The image is a frame of a movie or a plane of a Z-stack")

    def __init__(self):
        subpredicates = (self.IS_COLOR_PREDICATE,
                         self.IS_MONOCHROME_PREDICATE,
                         self.IS_STACK_PREDICATE,
                         self.IS_STACK_FRAME_PREDICATE)
        predicates = [pred_class(subpredicates, text)
                      for pred_class, text in (
                          (cellprofiler.setting.Filter.DoesPredicate, "Is"),
                          (cellprofiler.setting.Filter.DoesNotPredicate, "Is not"))]
        cellprofiler.setting.Filter.FilterPredicate.__init__(self,
                                            'image', "Image", self.fn_filter,
                                                             predicates,
                                                             doc="Filter based on image characteristics")

    def fn_filter(self, (node_type, modpath, module), *args):
        if node_type == cellprofiler.setting.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        ipd = module.get_image_plane_details(modpath)
        if ipd is None:
            return None
        return args[0](ipd, *args[1:])

    class FakeModule(cellprofiler.module.Module):
        """A fake module for setting validation"""

        def get_image_plane_details(self, modpath):
            url = cellprofiler.utilities.url.modpath_to_url(modpath)
            return cellprofiler.pipeline.ImagePlaneDetails(url, None, None, None)

    def test_valid(self, pipeline, *args):
        self((cellprofiler.setting.FileCollectionDisplay.NODE_FILE,
              ["/imaging", "test.tif"], self.FakeModule()), *args)


def is_image_extension(suffix):
    """Return True if the extension is one of those recongized by bioformats"""
    extensions = javabridge.get_collection_wrapper(
            javabridge.static_call("org/cellprofiler/imageset/filter/IsImagePredicate",
                          "getImageSuffixes", "()Ljava/util/Set;"))
    return extensions.contains(suffix.lower())