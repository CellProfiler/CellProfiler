pymodule_to_cpmodule = {
    "align": "Align",
    "groups": "Groups",
    "images": "Images",
    "loaddata": "LoadData",
    "metadata": "Metadata",
    "namesandtypes": "NamesAndTypes",
    "measurementfixture": "MeasurementFixture",
}
builtin_modules = [
    "align",
    "groups",
    "images",
    "loaddata",
    "metadata",
    "namesandtypes",
]
all_modules: dict = {}
svn_revisions: dict = {}
pymodules: list = []
badmodules: list = []
do_not_override = ["set_settings", "create_from_handles", "test_valid", "module_class"]
should_override = ["create_settings", "settings", "run"]
replaced_modules = {
    "LoadImageDirectory": ["LoadData"],
    "GroupMovieFrames": ["LoadData"],
    "IdentifyPrimLoG": ["IdentifyPrimaryObjects"],
    "FileNameMetadata": ["LoadData"],
    "LoadSingleImage": ["LoadData"],
    "LoadImages": ["LoadData"],
}
depricated_modules = ["CorrectIllumination_Calculate_kate", "SubtractBackground"]
unimplemented_modules = ["LabelImages", "Restart", "SplitOrSpliceMovie"]
