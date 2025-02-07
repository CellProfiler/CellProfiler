
ASSIGN_ALL = "All images"
ASSIGN_GUESS = "Try to guess image assignment"
ASSIGN_RULES = "Images matching rules"

LOAD_AS_GRAYSCALE_IMAGE = "Grayscale image"
LOAD_AS_COLOR_IMAGE = "Color image"
LOAD_AS_MASK = "Binary mask"
LOAD_AS_MASK_V5A = "Mask"
LOAD_AS_ILLUMINATION_FUNCTION = "Illumination function"
LOAD_AS_OBJECTS = "Objects"
LOAD_AS_ALL = [
    LOAD_AS_GRAYSCALE_IMAGE,
    LOAD_AS_COLOR_IMAGE,
    LOAD_AS_MASK,
    LOAD_AS_ILLUMINATION_FUNCTION,
    LOAD_AS_OBJECTS,
]

INTENSITY_RESCALING_BY_DATATYPE = "Image bit-depth"
INTENSITY_RESCALING_BY_METADATA = "Image metadata"
INTENSITY_MANUAL = "Manual"
INTENSITY_ALL = [
    INTENSITY_RESCALING_BY_DATATYPE,
    INTENSITY_RESCALING_BY_METADATA,
    INTENSITY_MANUAL,
]
MANUAL_INTENSITY_LABEL = "Maximum intensity"

RESCALING_HELP_TEXT = """\
This option determines how the image intensity should be rescaled from
0.0 – 1.0.

-  *{INTENSITY_RESCALING_BY_DATATYPE}:* Rescale the image to 0 – 1
   depending on the number of bits used to store the image
   (e.g. 255, 65535, etc.)
-  *{INTENSITY_RESCALING_BY_METADATA}:* Rescale the image intensity
   so that saturated values are rescaled to 1.0 by dividing all pixels
   in the image by the maximum possible intensity value allowed by the
   imaging hardware. Some image formats save the maximum possible
   intensity value along with the pixel data. For instance, a microscope
   might acquire images using a 12-bit A/D converter which outputs
   intensity values between zero and 4095, but stores the values in a
   field that can take values up to 65535.Choosing this setting will try its
   hardest to ensure that the intensity scaling value is the maximum
   allowed by the hardware, and not the maximum allowable by the file format.
   Although, this information is often unavailable, so may need to default
   to image datatype instead.
-  *{INTENSITY_MANUAL}:* Divide each pixel value by the value entered
   in the *{MANUAL_INTENSITY_LABEL}* setting. *{INTENSITY_MANUAL}*
   can be used to rescale an image whose maximum intensity metadata
   value is absent or incorrect, but is less than the value that would
   be supplied if *{INTENSITY_RESCALING_BY_DATATYPE}* were specified.

Please note that CellProfiler does not provide the option of loading the
image as the raw, unscaled values. If you wish to make measurements on
the unscaled image, use the **ImageMath** module to multiply the scaled
image by the actual image bit-depth.
""".format(
    **{
        "INTENSITY_RESCALING_BY_DATATYPE": INTENSITY_RESCALING_BY_DATATYPE,
        "INTENSITY_RESCALING_BY_METADATA": INTENSITY_RESCALING_BY_METADATA,
        "INTENSITY_MANUAL": INTENSITY_MANUAL,
        "MANUAL_INTENSITY_LABEL": MANUAL_INTENSITY_LABEL,
    }
)

MANUAL_RESCALE_HELP_TEXT = """\
*(Used only if “{INTENSITY_MANUAL}” is chosen)*

**NamesAndTypes** divides the pixel value, as read from the image file,
by this value to get the loaded image’s per-pixel intensity.
""".format(
    **{"INTENSITY_MANUAL": INTENSITY_MANUAL}
)

LOAD_AS_CHOICE_HELP_TEXT = """\
You can specify how these images should be treated:

-  *{LOAD_AS_GRAYSCALE_IMAGE}:* An image in which each pixel
   represents a single intensity value. Most of the modules in
   CellProfiler operate on images of this type.
   If this option is applied to a color image, the red, green and blue
   pixel intensities will be averaged to produce a single intensity
   value.
-  *{LOAD_AS_COLOR_IMAGE}:* An image in which each pixel represents a
   red, green and blue (RGB) triplet of intensity values OR which contains
   multiple individual grayscale channels. Please note
   that the object detection modules such as **IdentifyPrimaryObjects**
   expect a grayscale image, so if you want to identify objects, you
   should use the **ColorToGray** module in the analysis pipeline to
   split the color image into its component channels.
   You can use the **ColorToGray**'s *Combine* option after image loading
   to collapse the color channels to a single grayscale value if you don’t need
   CellProfiler to treat the image as color.
-  *{LOAD_AS_MASK}:* A *mask* is an image where some of the pixel
   intensity values are zero, and others are non-zero. The most common
   use for a mask is to exclude particular image regions from
   consideration. By applying a mask to another image, the portion of
   the image that overlaps with the non-zero regions of the mask are
   included. Those that overlap with the zeroed region are “hidden” and
   not included in downstream calculations. For this option, the input
   image should be a binary image, i.e, foreground is white, background
   is black. The module will convert any nonzero values to 1, if needed.
   You can use this option to load a foreground/background segmentation
   produced by the **Threshold** module or one of the **Identify** modules.
-  *{LOAD_AS_ILLUMINATION_FUNCTION}:* An *illumination correction
   function* is an image which has been generated for the purpose of
   correcting uneven illumination/lighting/shading or to reduce uneven
   background in images. Typically, is a file in the NumPy .npy format.
   See **CorrectIlluminationCalculate** and **CorrectIlluminationApply**
   for more details.
-  *{LOAD_AS_OBJECTS}:* Use this option if the input image is a label
   matrix and you want to obtain the objects that it defines. A label
   matrix is a grayscale or color image in which the connected regions
   share the same label, which defines how objects are represented in
   CellProfiler. The labels are integer values greater than or equal to
   0. The elements equal to 0 are the background, whereas the elements
   equal to 1 make up one object, the elements equal to 2 make up a
   second object, and so on. This option allows you to use the objects
   immediately without needing to insert an **Identify** module to
   extract them first. See **IdentifyPrimaryObjects** for more details.
   This option can load objects created by using the **ConvertObjectsToImage**
   module followed by the **SaveImages** module. Loaded objects can take two
   forms, with different considerations for each:

   -  *Non-overlapping* objects are stored as a label matrix. This
      matrix should be saved as grayscale rather than color.
   -  *Overlapping objects* are stored in a multi-frame TIF, each frame
      of which consists of a grayscale label matrix. The frames are
      constructed so that objects that overlap are placed in different
      frames. CellProfiler currently does not support saving of overlapping
      objects, so these can only be used within the pipeline.
""".format(
    **{
        "LOAD_AS_COLOR_IMAGE": LOAD_AS_COLOR_IMAGE,
        "LOAD_AS_GRAYSCALE_IMAGE": LOAD_AS_GRAYSCALE_IMAGE,
        "LOAD_AS_ILLUMINATION_FUNCTION": LOAD_AS_ILLUMINATION_FUNCTION,
        "LOAD_AS_MASK": LOAD_AS_MASK,
        "LOAD_AS_OBJECTS": LOAD_AS_OBJECTS,
    }
)

IDX_ASSIGNMENTS_COUNT_V2 = 5
IDX_ASSIGNMENTS_COUNT_V3 = 6
IDX_ASSIGNMENTS_COUNT_V5 = 6
IDX_ASSIGNMENTS_COUNT_V6 = 6
IDX_ASSIGNMENTS_COUNT_V7 = 6
IDX_ASSIGNMENTS_COUNT = 6

IDX_SINGLE_IMAGES_COUNT_V5 = 7
IDX_SINGLE_IMAGES_COUNT_V6 = 7
IDX_SINGLE_IMAGES_COUNT_V7 = 7
IDX_SINGLE_IMAGES_COUNT = 7

IDX_FIRST_ASSIGNMENT_V3 = 7
IDX_FIRST_ASSIGNMENT_V4 = 7
IDX_FIRST_ASSIGNMENT_V5 = 8
IDX_FIRST_ASSIGNMENT_V6 = 9
IDX_FIRST_ASSIGNMENT_V7 = 13
IDX_FIRST_ASSIGNMENT = 13

NUM_ASSIGNMENT_SETTINGS_V2 = 4
NUM_ASSIGNMENT_SETTINGS_V3 = 5
NUM_ASSIGNMENT_SETTINGS_V5 = 7
NUM_ASSIGNMENT_SETTINGS_V6 = 8
NUM_ASSIGNMENT_SETTINGS_V7 = 8
NUM_ASSIGNMENT_SETTINGS = 6

NUM_SINGLE_IMAGE_SETTINGS_V5 = 7
NUM_SINGLE_IMAGE_SETTINGS_V6 = 8
NUM_SINGLE_IMAGE_SETTINGS_V7 = 8
NUM_SINGLE_IMAGE_SETTINGS = 6


OFF_LOAD_AS_CHOICE_V5 = 3
OFF_LOAD_AS_CHOICE = 3

OFF_SI_LOAD_AS_CHOICE_V5 = 3
OFF_SI_LOAD_AS_CHOICE = 3

MATCH_BY_ORDER = "Order"
MATCH_BY_METADATA = "Metadata"

IMAGE_NAMES = ["DNA", "GFP", "Actin"]
OBJECT_NAMES = ["Cell", "Nucleus", "Cytoplasm", "Speckle"]

DEFAULT_MANUAL_RESCALE = 255

"""The experiment measurement that holds the ZLIB compression dictionary for image sets"""
M_IMAGE_SET_ZIP_DICTIONARY = "ImageSet_Zip_Dictionary"
"""The image measurement that holds the compressed image set"""
M_IMAGE_SET = "ImageSet_ImageSet"

# Image set error types
E_WRONG_LENGTH = "missing images"
E_MISSING = "no matches"
E_TOO_MANY = "too many matches"

