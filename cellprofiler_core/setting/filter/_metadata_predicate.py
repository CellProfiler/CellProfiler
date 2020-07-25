import cellprofiler_core.pipeline
import cellprofiler_core.setting
from cellprofiler_core.modules.namesandtypes import NamesAndTypes


class MetadataPredicate(cellprofiler_core.setting.Filter.FilterPredicate):
    """A predicate that compares an ifd against a metadata key and value"""

    SYMBOL = "metadata"

    def __init__(self, display_name, display_fmt="%s", **kwargs):
        subpredicates = [
            cellprofiler_core.setting.Filter.DoesPredicate([]),
            cellprofiler_core.setting.Filter.DoesNotPredicate([]),
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
            cellprofiler_core.setting.Filter.FilterPredicate(
                key,
                self.display_fmt % key,
                lambda ipd, match, key=key: key in ipd.metadata
                and ipd.metadata[key] == match,
                [cellprofiler_core.setting.Filter.LITERAL_PREDICATE],
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
        node_type, modpath, resolver = arg
        ipd = resolver.get_image_plane_details(modpath)
        return vargs[0](ipd, *vargs[1:])

    def test_valid(self, pipeline, *args):
        modpath = ["imaging", "image.png"]
        ipd = cellprofiler_core.pipeline.ImagePlane(
            "/imaging/image.png", None, None, None
        )
        self(
            (
                cellprofiler_core.setting.FileCollectionDisplay.NODE_IMAGE_PLANE,
                modpath,
                NamesAndTypes.FakeModpathResolver(modpath, ipd),
            ),
            *args,
        )
