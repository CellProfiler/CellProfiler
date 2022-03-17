from ._does_not_predicate import DoesNotPredicate
from ._does_predicate import DoesPredicate
from ._filter import LITERAL_PREDICATE
from ._filter_predicate import FilterPredicate
from .._file_collection_display import FileCollectionDisplay
from ...pipeline import ImageFile
from ...pipeline import ImagePlane as ImagePlane


class MetadataPredicate(FilterPredicate):
    """A predicate that compares an ifd against a metadata key and value"""

    SYMBOL = "metadata"

    def __init__(self, display_name, display_fmt="%s", **kwargs):
        subpredicates = [
            DoesPredicate([]),
            DoesNotPredicate([]),
        ]

        super(self.__class__, self).__init__(
            self.SYMBOL,
            display_name,
            MetadataPredicate.do_filter,
            subpredicates,
            **kwargs,
        )
        self.display_fmt = display_fmt

    def set_metadata_keys(self, keys):
        """Define the possible metadata keys to be matched against literal values

        keys - a list of keys
        """
        sub_subpredicates = [
            FilterPredicate(
                key,
                self.display_fmt % key,
                lambda plane, match, key=key: str(plane.get_metadata(key)) == match,
                [LITERAL_PREDICATE],
            )
            for key in keys
        ]
        #
        # The subpredicates are "Does" and "Does not", so we add one level
        # below that.
        #
        for subpredicate in self.subpredicates:
            subpredicate.subpredicates = sub_subpredicates

    @classmethod
    def do_filter(cls, arg, *vargs):
        """Perform the metadata predicate's filter function

        The metadata predicate has subpredicates that look up their
        metadata key in the ipd and compare it against a literal.
        """
        node_type, modpath, plane = arg
        return vargs[0](plane, *vargs[1:])

    def test_valid(self, pipeline, *args):
        print("Testing validity of a metadata predicate using a fake image plane")

        class FakeModpathResolver(object):
            """Resolve one modpath to one ipd"""

            def __init__(self, modpath, ipd):
                self.modpath = modpath
                self.ipd = ipd

            def get_image_plane_details(self, modpath):
                assert len(modpath) == len(self.modpath)
                assert all([m1 == m2 for m1, m2 in zip(self.modpath, modpath)])
                return self.ipd

        modpath = ["imaging", "image.png"]
        plane = ImagePlane(ImageFile("/imaging/image.png"), None, None, None)
        self(
            (
                FileCollectionDisplay.NODE_IMAGE_PLANE,
                modpath,
                FakeModpathResolver(modpath, plane),
            ),
            *args,
        )
