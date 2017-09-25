Using the Data Tools Menu
=========================

The *Data Tools* menu provides tools to allow you to plot, view, export
or perform specialized analyses on your measurements.

Each data tool has a corresponding module with the same name and
functionality. The difference between the data tool and the module is
that the data tool takes a CellProfiler output file (i.e., a *.mat or
.h5* file) as input, which contains measurements from a previously
completed analysis run. In contrast, a module uses measurements received
from the upstream modules during an in-progress analysis run.

Opening a data tool will present a prompt in which the user is asked to
provide the location of the output file. Once specified, the user is
then prompted to enter the desired settings. The settings behave
identically as those from the corresponding module.

Please note that with the exception of *PlateViewer* and *Export* functions the
*Data Tools*, like most CellProfiler modules, are designed to operate on only one image
set at a time. If you want to use data tool modules to examine and/or
graph data on the whole experiment level, you should instead consider using
CellProfiler Analyst; see the *ExportToDatabase* help to learn more about exporting
your data into a database that CellProfiler Analyst can access and about creating a
CellProfiler Analyst properties file.

Help for each *Data Tool* is available under *Data Tools > Help*
or the corresponding module help.
