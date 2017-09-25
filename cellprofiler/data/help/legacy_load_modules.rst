Load Modules
============

The image loading modules **LoadImages** and **LoadSingleImage** are deprecated
and will be removed in a future version of CellProfiler. It is recommended you
choose to convert these modules as soon as possible. CellProfiler can do this
automatically for you when you import a pipeline using either of these legacy
modules.

Historically, these modules served the same functionality as the current
project structure (via **Images**, **Metadata**, **NamesAndTypes**, and **Groups**).
Pipelines loaded into CellProfiler that contain these modules will provide the option
of preserving them; these pipelines will operate exactly as before.

The section details information relevant for those who would like
to continue using these modules. Please note, however, that these
modules are deprecated and will be removed in a future version of CellProfiler.

Associating metadata with images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Metadata (i.e., additional data about image data) is sometimes available
for input images. This information can be:

#. Used by CellProfiler to group images with common metadata identifiers
   (or “tags”) together for particular steps in a pipeline;
#. Stored in the output file along with CellProfiler-measured features
   for annotation or sample-tracking purposes;
#. Used to name additional input/output files.

Metadata is provided in the image filename or location (pathname). For
example, images produced by an automated microscope can be given
names such as “Experiment1\_A01\_w1\_s1.tif” in which the metadata
about the plate (“Experiment1”), the well (“A01”), the wavelength
number (“w1”) and the imaging site (“s1”) are captured. The name
of the folder in which the images are saved may be meaningful and may
also be considered metadata as well. If this is the case for your
data, use **LoadImages** to extract this information for use in the
pipeline and storage in the output file.

Details for the metadata-specific help is given next to the appropriate
settings in **LoadImages**, as well the specific
settings in other modules which can make use of metadata. However, here
is an overview of how metadata is obtained and used.

In **LoadImages**, metadata can be extracted from the filename and/or
folder location using regular expression, a specialized syntax used for
text pattern-matching. These regular expressions can be used to identify
different parts of the filename / folder. The syntax
*(?P<fieldname>expr)* will extract whatever matches *expr* and assign it
to the image’s *fieldname* measurement. A regular expression tool is
available which will allow you to check the accuracy of your regular
expression.

For instance, say a researcher has folder names with the date and
subfolders containing the images with the run ID (e.g.,
*./2009\_10\_02/1234/*). The following regular expression will capture
the plate, well and site in the fields *Date* and *Run*:
``.\*[\\\\\\/](?P<Date>.\\*)[\\\\\\\\/](?P<Run>.\\*)$``

================   ============
Subexpression      Explanation
================   ============
.\\*[\\\\\\\\/]      Skip characters at the beginning of the pathname until either a slash (/) or backslash (\\\\) is encountered (depending on the OS). The extra slash for the backslash is used as an escape sequence.
(?P<Date>          Name the captured field *Date*
.\\*                Capture as many characters that follow
[\\\\\\\\/]            Discard the slash/backslash character
(?P<Run>           Name the captured field *Run*
$                  The *Run* field must be at the end of the path string, i.e., the last folder on the path. This also means that the *Date* field contains the parent folder of the *Date* folder.
================   ============

In **LoadImages**, metadata is extracted from the image *File name*,
*Path* or *Both*. File names or paths containing “Metadata” can be used
to group files loaded by **LoadImages** that are associated with a common
metadata value. The files thus grouped together are then processed as a
distinct image set.

For instance, an experiment might require that images created on the
same day use an illumination correction function calculated from all
images from that day, and furthermore, that the date be captured in the
file names for the individual image sets specifying the illumination
correction functions.

In this case, if the illumination correction images are loaded with the
**LoadImages** module, **LoadImages** should be set to extract the metadata
tag from the file names. The pipeline will then match the individual images
with their corresponding illumination correction functions based on matching
“Metadata\_Date” fields.

Using image grouping
~~~~~~~~~~~~~~~~~~~~

To use grouping, you must define the relevant metadata for each image.
This can be done using regular expressions in **LoadImages**.

To use image grouping in **LoadImages**, please note the following:

-  *Metadata tags must be specified for all images listed.* You cannot
   use grouping unless an appropriate regular expression is defined for
   all the images listed in the module.
-  *Shared metadata tags must be specified with the same name for each
   image listed.* For example, if you are grouping on the basis of a
   metadata tag “Plate” in one image channel, you must also specify the
   “Plate” metadata tag in the regular expression for the other channels
   that you want grouped together.