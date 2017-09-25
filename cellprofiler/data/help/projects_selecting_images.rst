Selecting Images for Input
==========================

Any image analysis project using CellProfiler begins with providing the
program with a set of image files to be analyzed. You can do this by
clicking on the **Images** module to select it (located in the Input
modules panel on the left); this module is responsible for collecting
the names and locations of the files to be processed.

The most straightforward way to provide files to the **Images** module
is to simply drag-and-drop them from your file manager tool (e.g.,
Windows Explorer, Mac Finder) onto the file list panel (the blank space
indicated by the text “Drop files and folders here”). Both individual
files and entire folders can be dragged onto this panel, and as many
folders and files can be placed onto this panel as needed. As you add
files, you will see a listing of the files appear in the panel.

CellProfiler supports a wide variety of image formats, including most of
those used in imaging, by using a library called Bio-Formats; see
`here`_ for the formats available. Some image formats are better than
others for image analysis. Some are `“lossy”`_ (information is lost in
the conversion to the format) like most JPG/JPEG files; others are
`lossless`_ (no image information is lost). For image analysis purposes,
a lossless format like TIF or PNG is recommended.

If you have a subset of files that you want to analyze from the full
list shown in the panel, you can also filter the files according to a
set of rules that you specify. This is useful when, for example, you
have dragged a folder of images onto the file list panel, but the folder
contains the images from one experiment that you want to process along
with images from another experiment that you want to ignore for now. You
may specify as many rules as necessary to define the desired list of
images.

For more information on this module and how to configure it for the best
performance, please see the detailed help by selecting the module and
clicking the |image0| button at the bottom of the pipeline panel, or
check out the Input module tutorials on our `Tutorials`_ page.

.. _here: http://docs.openmicroscopy.org/bio-formats/5.7.0/supported-formats.html
.. _“lossy”: http://www.techterms.com/definition/lossy
.. _lossless: http://www.techterms.com/definition/lossless
.. _Tutorials: http://cellprofiler.org/tutorials/

.. |image0| image:: ../images/module_help.png
