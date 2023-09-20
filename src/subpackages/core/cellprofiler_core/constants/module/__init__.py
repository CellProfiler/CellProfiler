from cellprofiler_core.preferences import (
    ABSOLUTE_FOLDER_NAME,
    DEFAULT_INPUT_SUBFOLDER_NAME,
    DEFAULT_OUTPUT_SUBFOLDER_NAME,
)
from cellprofiler_core.utilities.image import image_resource

DEFINITION_OBJECT = """\
In CellProfiler, we use the term *object* as a generic term to refer to
an identified feature in an image, usually an organism, cell, or cellular
compartment (for example, nuclei, cells, colonies, worms).
"""
REGEXP_HELP_REF = """\
**Details on regular expressions:**

A *regular expression* is a general term referring to a method of
searching for pattern matches in text. There is a high learning curve to
using them, but are quite powerful once you understand the basics.

Patterns are specified using combinations of metacharacters and literal
characters. There are a few classes of metacharacters, partially listed
below. Some helpful links follow:

-  A more extensive explanation of regular expressions can be found
   `here`_
-  A helpful quick reference can be found
   `here <http://www.cheatography.com/davechild/cheat-sheets/regular-expressions/>`__
-  `Pythex`_ provides quick way to test your regular expressions. Here
   is an `example`_ to capture information from a common microscope
   nomenclature.

.. _here: http://docs.python.org/2/howto/regex.html
.. _Pythex: http://pythex.org/
.. _example: http://pythex.org/?regex=Channel%5B1-2%5D-%5B0-9%5D%7B2%7D-(%3FP%3CWellRow%3E%5BA-H%5D)-(%3FP%3CWellColumn%3E%5B0-9%5D%7B2%7D)%5C.tif&test_string=Channel1-01-A-01.tif&ignorecase=0&multiline=0&dotall=0&verbose=0
"""
FILTER_RULES_BUTTONS_HELP = """\
Clicking the rule menus shows you all the file *attributes*, *operators*
and *conditions* you can specify to narrow down the image list.

#. For each rule, first select the *attribute* that the rule is to be
   based on. For example, you can select “File” to define a rule that
   will filter files on the basis of their filename.
#. The *operator* drop-down is then updated with operators applicable to
   the attribute you selected. For example, if you select “File” as the
   attribute, the operator menu includes text operators such as
   *Contain* or *Starts with*. On the other hand, if you select
   “Extension” as the attribute, you can choose the logical operators
   “Is” or “Is not” from the menu.
#. In the operator drop-down menu, select the operator you want to use.
   For example, if you want to match data exactly, you may want the
   “Exactly match” or the “Is” operator. If you want the condition to be
   more loose, select an operator such as “Contains”.
#. Use the *condition* box to type the condition you want to match. The
   more you type, the more specific the condition is.

   -  As an example, if you create a new filter and select *File* as the
      attribute, then select “Does” and “Contain” as the operators, and
      type “BBBC013” as the condition, the filter finds all files that
      include the text “BBBC013”, such as “BBBC013-1.tif” “BBBC013-2.jpg”,
      “1-BBBC013-A01.BMP” and so on.
   -  If you select “Does” and “Start with” as the operators and
      “BBBC013-1” in the Condition box, the rule will includes such files
      as “BBBC013-1.tif” “BBBC013-1-A01.png”, and so on.

.. image:: {IMAGES_USING_RULES_ICON}
   :width: 100%

You can also create regular expressions (an advanced syntax for
pattern matching) in order to select particular files.

To add another rule, click the plus buttons to the right of each rule.
Subtract an existing rule by clicking the minus button.

You can also link a set of rules by choosing the logical expression
*All* or *Any*. If you use *All* logical expression, all the rules must
be true for a file to be included in the File list. If you use the *Any*
option, only one of the conditions has to be met for a file to be
included.

If you want to create more complex rules (e.g, some criteria matching
all rules and others matching any), you can create sets of rules, by
clicking the ellipsis button (to the right of the plus button). Repeat
the above steps to add more rules to the filter until you have all the
conditions you want to include.

{REGEXP_HELP_REF}
""".format(
    **{
        "IMAGES_USING_RULES_ICON": image_resource("Images_UsingRules.png"),
        "REGEXP_HELP_REF": REGEXP_HELP_REF,
    }
)
HELP_ON_MEASURING_DISTANCES = """\
To measure distances in an open image, use the “Measure length” tool
under *Tools* in the display window menu bar. If you click on an image
and drag, a line will appear between the two endpoints, and the distance
between them will be shown at the right-most portion of the bottom panel.\
"""
HELP_ON_MEASURING_INTENSITIES = """\
Note that for publication purposes, the units of intensity from
microscopy images are usually described as “Intensity units” or
“Arbitrary intensity units” because microscopes are not calibrated to an
absolute scale. Also, it is important to note whether you are reporting
the mean vs. the integrated intensity, so specify “Mean intensity
units” or “Integrated intensity units” accordingly.

Keep in mind that the default behavior in CellProfiler is to rescale the
image intensity from 0 to 1 by dividing all pixels in the image by the
maximum possible intensity value. This “maximum possible” value is
defined by the “Set intensity range from” setting in **NamesAndTypes**;
see the help for that setting for more details.
"""
HELP_ON_PIXEL_INTENSITIES = """\
To view pixel intensities in an open image, use the pixel intensity tool
which is available in any open display window. When you move your mouse
over the image, the pixel intensities will appear in the bottom bar of
the display window.\
"""
IO_FOLDER_CHOICE_HELP_TEXT = """\
You can choose among the following options which are common to all file
input/output modules:

-  *Default Input Folder*: Use the default input folder.
-  *Default Output Folder:* Use the default output folder.
-  *Elsewhere…*: Use a particular folder you specify.
-  *Default input directory sub-folder*: Enter the name of a subfolder
   of the default input folder or a path that starts from the default
   input folder.
-  *Default output directory sub-folder*: Enter the name of a subfolder
   of the default output folder or a path that starts from the default
   output folder.

*Elsewhere* and the two sub-folder options all require you to enter an
additional path name. You can use an *absolute path* (such as
“C:\\\\imagedir\\\\image.tif” on a PC) or a *relative path* to specify
the file location relative to a directory, which makes the pipeline
more flexible for future runs):

-  Use one period to represent the current directory. For example, if
   you choose *Default Input Folder sub-folder*, you can enter
   “./MyFiles” to look in a folder called “MyFiles” that is contained
   within the Default Input Folder.
-  Use two periods “..” to move up one folder level. For example, if you
   choose *Default Input Folder sub-folder*, you can enter “../MyFolder”
   to look in a folder called “MyFolder” at the same level as the
   Default Input Folder.\
"""
USING_METADATA_GROUPING_HELP_REF = """\
Please see the **Groups** module for more details on the proper use of
metadata for grouping.
"""
USING_METADATA_HELP_REF = """\
Please see the **Metadata** module for more details on metadata
collection and usage.
"""
USING_METADATA_TAGS_REF = """\
You can insert a previously defined metadata tag by either using:

-  The insert key
-  A right mouse button click inside the control
-  In Windows, the Context menu key, which is between the Windows key
   and Ctrl key

The inserted metadata tag will appear in green. To change a previously
inserted metadata tag, navigate the cursor to just before the tag and
either:

-  Use the up and down arrows to cycle through possible values.
-  Right-click on the tag to display and select the available values.
"""
IO_WITH_METADATA_HELP_TEXT = """\
For *{ABSOLUTE_FOLDER_NAME}*, *{DEFAULT_INPUT_SUBFOLDER_NAME}* and
*{DEFAULT_OUTPUT_SUBFOLDER_NAME}*, if you have metadata associated
with your images via **Metadata** module, you can name the folder using any
metadata tags for which all images in each individual image set have the same value.

-  Example: if you had extracted "*Plate*", "*Well*", and "*Channel*" metadata
   from your images, for most pipelines folders based on "*Plate*" or "*Well*" would work since
   each individual image set would come only from a single well on a single plate, but
   folders based on "*Channel*" would not work as each individual image set might
   contain many channels.

{USING_METADATA_TAGS_REF}

For instance, if you have a metadata tag named “Plate”, you can create a
per-plate folder by selecting one of the subfolder options and then
specifying the subfolder name as “\\g<Plate>”. The module will
substitute the metadata values for the current image set for any
metadata tags in the folder name.

{USING_METADATA_HELP_REF}
""".format(
    **{
        "ABSOLUTE_FOLDER_NAME": ABSOLUTE_FOLDER_NAME,
        "DEFAULT_INPUT_SUBFOLDER_NAME": DEFAULT_INPUT_SUBFOLDER_NAME,
        "DEFAULT_OUTPUT_SUBFOLDER_NAME": DEFAULT_OUTPUT_SUBFOLDER_NAME,
        "USING_METADATA_HELP_REF": USING_METADATA_HELP_REF,
        "USING_METADATA_TAGS_REF": USING_METADATA_TAGS_REF,
    }
)
HELP_ON_SAVING_OBJECTS = """\
*Note on saving images:* You can pass the objects along to the
*Object Processing* module **ConvertObjectsToImage** to create an image.
This image can be saved with the **SaveImages** module. Additionally,
you can use the **OverlayOutlines** or **OverlayObjects** module to
overlay outlines or objects, respectively, on a base image.
The resulting image can also be saved with the **SaveImages** module.
"""
StrelImage = image_resource("structuringelement.png")
HELP_FOR_STREL = """\
The structuring element is the shape that will be applied in any morphological
operation. The structuring element is centered on each pixel and the shape and size
selected will determine what neighborhood around that pixel will be affected by
that operation. See image below for an example of dilating a starting square of 11
pixel diameter with various structuring elements.

|StrelImage|

.. |StrelImage| image:: {StrelImage}
""".format(
    **{"StrelImage": StrelImage}
)
PROTIP_RECOMMEND_ICON = image_resource("thumb-up.png")
PROTIP_AVOID_ICON = image_resource("thumb-down.png")
TECH_NOTE_ICON = image_resource("gear.png")
