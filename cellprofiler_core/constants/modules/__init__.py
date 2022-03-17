builtin_modules = {
    "align": "Align",
    "groups": "Groups",
    "images": "Images",
    "loaddata": "LoadData",
    "metadata": "Metadata",
    "namesandtypes": "NamesAndTypes",
}
all_modules: dict = {}
svn_revisions: dict = {}
pymodules: list = []
badmodules: list = []
do_not_override = ["set_settings", "create_from_handles", "test_valid", "module_class"]
should_override = ["create_settings", "settings", "run"]
renamed_modules = {
    "ApplyThreshold": "Threshold",
    "CalculateImageOverlap": "MeasureImageOverlap",
    "CropObjects": "SaveCroppedObjects",
    "Dilation": "DilateImage",
    "MeasureNeurons": "MeasureObjectSkeleton",
    "MeasureCorrelation": "MeasureColocalization",
    "NoiseReduction": "ReduceNoise",
    "ReassignObjectNumbers": "SplitOrMergeObjects",
    "SplitIntoContiguousObjects": "SplitOrMergeObjects",
    "Erosion": "ErodeImage",
}
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
