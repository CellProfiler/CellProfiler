Introduction to Projects
========================

What is a project?
~~~~~~~~~~~~~~~~~~

In CellProfiler, a *project* is comprised of two elements:

-  An *image file list* which is the list of files and their locations
   that you select as candidates for analysis.
-  The *pipeline*, which is a series of modules put together to
   analyze a set of images.
-  Optionally, the associated information about the images (*metadata*).
   This information may be extracted from the images themselves, or you may
   import them from an external file.

The project is the container for image information associated with a
CellProfiler analysis. It stores such details as:

-  What type of image(s) are the input files?
-  Where are the input images located?
-  What distinguishes multiple image channels from each other? How are
   these relationships represented?
-  What information about the images and/or experiment is linked to the
   images, and how?
-  Are certain groups of images to be processed differently from other
   groups?


Working with projects
~~~~~~~~~~~~~~~~~~~~~

Creating a project
^^^^^^^^^^^^^^^^^^

Upon starting CellProfiler, you will be presented with a new, blank
project. At this point, you may start building your project by using the
modules located in the pipeline window on the upper-left. The
modules are:

-  **Images**: Assemble the relevant images for analysis (required).
-  **Metadata**: Associate metadata with the images (optional).
-  **NamesAndTypes**: Assign names to channels and define their
   relationship (required).
-  **Groups**: Define sub-divisions between groups of images for
   processing (optional).

Detailed help for each module is provided by selecting the module and
clicking the “?” button on the bottom of CellProfiler.

Rather than the four input modules, you may alternatively use the **LoadData**
module to associate images with your project.  **LoadData** allows you to upload
a spreadsheet to provide the same location, metadata, channel, and grouping information 
as the four input modules.  See the help for that module for more information.

Saving a project
^^^^^^^^^^^^^^^^

As you work in CellProfiler, the project is updated automatically, so
there is no need to save it unless you are saving the project to a new
name or location. You can always save your current work to a new project
file by selecting *File > Save Project As…*, which will save your
project, complete with the current image file list and pipeline, to a
file with the extension *.cpproj*.

As an alternative, you can save the pipeline by itself to a file with the extention *.cppipe*, by selecting *File > Export > Pipeline*. *.cppipe* files contain only the list of steps and their corresponding settings and do not contain information about images.

You also have the option of automatically saving the associated pipeline
file and the file list in addition to the project file. See *File >
Preferences…* for more details.

For those interested, some technical details:

-  The *.cpproj* file stores collected information using the HDF5
   format. Documentation on how measurements are stored and handled in
   CellProfiler using this format can be found `here`_.
-  All information is cached in the project file after it is computed.
   It is either re-computed or retrieved from the cache when an analysis
   run is started, when entering Test mode, or when the user requests a
   refreshed view of the information (e.g., when a setting has been
   changed).

.. _here: http://github.com/CellProfiler/CellProfiler/wiki/Module-structure-and-data-storage-retrieval#HDF5
