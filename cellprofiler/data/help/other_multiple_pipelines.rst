Running Multiple Pipelines
==========================

The **Run multiple pipelines** dialog lets you select several
pipelines which will be run consecutively. Please note the following:

-  Pipeline files (.cppipe) are supported.
-  Project files (.cpproj) from CellProfiler 2.1 or newer are not supported.
   To convert your project to a pipeline (.cppipe), select *File > Export > Pipeline...*
   and, under the “Save as type” dropdown, select “CellProfiler pipeline and file list”
   to export the project file list with the pipeline.

You can invoke **Run multiple pipelines** by selecting it from the file menu. The dialog has three parts to it:

-  *File chooser*: The file chooser lets you select the pipeline files
   to be run. The *Select all* and *Deselect all* buttons to the right
   will select or deselect all pipeline files in the list. The *Add*
   button will add the pipelines to the pipeline list. You can add a
   pipeline file multiple times, for instance if you want to run that
   pipeline on more than one input folder.
-  *Directory chooser*: The directory chooser lets you navigate to
   different directories. The file chooser displays all pipeline files
   in the directory chooser’s current directory.
-  *Pipeline list*: The pipeline list has the pipelines to be run in the
   order that they will be run. Each pipeline has a default input and
   output folder and a measurements file. You can change any of these by
   clicking on the file name - an appropriate dialog will then be
   displayed. You can click the remove button to remove a pipeline from
   the list.

CellProfiler will run all of the pipelines on the list when you hit
the “OK” button.
