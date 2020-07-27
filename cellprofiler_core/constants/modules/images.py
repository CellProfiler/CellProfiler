FILTER_CHOICE_NONE = "No filtering"
FILTER_CHOICE_IMAGES = "Images only"
FILTER_CHOICE_CUSTOM = "Custom"
FILTER_CHOICE_ALL = [FILTER_CHOICE_NONE, FILTER_CHOICE_IMAGES, FILTER_CHOICE_CUSTOM]
FILTER_DEFAULT = (
    'and (extension does isimage) (directory doesnot containregexp "[\\\\\\\\/]\\\\.")'
)
