STYLE_NO_MATCH = 0
STYLE_MATCH = 1
STYLE_FIRST_LABEL = 2
STYLE_ERROR = 31
UUID_REGEXP = (
    "[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
RE_FILENAME_GUESSES = [
    # This is the generic naming convention for fluorescent microscopy images
    "^(?P<Plate>.*?)_(?P<Well>[A-Za-z]+[0-9]+)f(?P<Site>[0-9]{2})d(?P<Dye>[0-9])\\.tif$",
    # Molecular devices single site
    "^(?P<ExperimentName>.*?)_(?P<Well>[A-Za-z]+[0-9]+)_w(?P<Wavelength>[0-9])_?"
    + UUID_REGEXP
    + "\\.tif$",
    # Plate / well / site / channel without UUID
    "^(?P<Plate>.*?)_(?P<Well>[A-Za-z]+[0-9]+)_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])\\.tif$",
    # Molecular devices multi-site
    "^(?P<ExperimentName>.*?)_(?P<Well>[A-Za-z]+[0-9]+)_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
    + UUID_REGEXP
    + "\\.tif$",
    # Molecular devices multi-site, single wavelength
    "^(?P<ExperimentName>.*)_(?P<Well>[A-Za-z][0-9]{2})_s(?P<Site>[0-9])" + UUID_REGEXP,
    # Plate / well / [UUID]
    "^(?P<Plate>.*?)_(?P<Well>[A-Za-z]+[0-9]+)_\\[" + UUID_REGEXP + "\\]\\.tif$",
    # Cellomics
    "^(?P<ExperimentName>.*)_(?P<Well>[A-Za-z][0-9]{1,2})f(?P<Site>[0-9]{1,2})d(?P<Wavelength>[0-9])",
    # BD Pathway
    "^(?P<Wavelength>.*) - n(?P<StackSlice>[0-9]{6})",
    # GE InCell Analyzer
    r"^(?P<Row>[A-H]*) - (?P<Column>[0-9]*)\(fld (?P<Site>[0-9]*) wv (?P<Wavelength>.*) - (?P<Filter>.*)\)",
    # Phenix
    r"^r(?P<WellRow>\d{2})c(?P<WellColumn>\d{2})f(?P<Site>\d{2})p\d{2}-ch(?P<ChannelNumber>\d)",
    # GE InCell Analyzer 7.2
    r"^(?P<Row>[A-P])_(?P<Column>[0-9]*)_f(?P<Site>[0-9]*)_c(?P<Channel>[0-9]*)_x(?P<Wavelength>.*)_m("
    r"?P<Filter>.*)_z(?P<Slice>[0-9]*)_t(?P<Timepoint>[0-9]*)\.tif",
    # Please add more guesses below
]
RE_FOLDER_GUESSES = [
    # BD Pathway
    r".*[\\/](?P<Plate>[^\\/]+)[\\/](?P<Well>[A-Za-z][0-9]{2})",
    # Molecular devices
    r".*[\\/](?P<Date>\d{4}-\d{1,2}-\d{1,2})[\\/](?P<PlateID>.*)$"
    # Please add more guesses below
]
TOK_ORDINARY = 0
TOK_ESCAPE = 1
TOK_GROUP = 2
TOK_BRACKET_EXP = 3
TOK_REPEAT = 4
TOK_SPECIAL = 5
TOK_DEFINITION = 6
HARDCODE_ESCAPES = {
    r"\\",
    r"\a",
    r"\b",
    r"\d",
    r"\f",
    r"\n",
    r"\r",
    r"\s",
    r"\t",
    r"\v",
    r"\w",
    r"\A",
    r"\B",
    r"\D",
    r"\S",
    r"\W",
    r"\Z",
}
OCTAL_DIGITS = set("01234567")
DECIMAL_DIGITS = set("0123456789")
HEXIDECIMAL_DIGITS = set("0123456789ABCDEFabcdef")
REPEAT_STARTS = set("{*+?")
OTHER_SPECIAL_CHARACTERS = set(".|")
IGNORABLE_GROUPS = (r"\(\?[iLmsux]+\)", r"\(\?#.*\)")
