Setting the Output Filename
===========================

Please note that the output file will be deprecated in the future. This
setting is temporarily present for those needing HDF5 or MATLAB formats,
and will be moved to Export modules in future versions of CellProfiler.

The *output file* is a file where all information about the analysis as
well as any measurements will be stored to the hard drive. **Important
note:** This file does *not* provide the same functionality as the
Export modules. If you want to produce a spreadsheet of measurements
easily readable by Excel or a database viewer (or similar programs),
please refer to the **ExportToSpreadsheet** or **ExportToDatabase**
modules and the associated help.

The options associated with the output file are accessible by pressing
the “View output settings” button at the bottom of the pipeline panel.
In the settings panel to the left, in the *Output Filename* box, you can
specify the name of the output file.

The output file can be written in one of two formats:

-  A *.mat file* which is readable by CellProfiler and by `MATLAB`_
   (Mathworks).
-  An *.h5 file* which is readable by CellProfiler, MATLAB and any other
   program capable of reading the HDF5 data format. Documentation on how
   measurements are stored and handled in CellProfiler using this format
   can be found `here`_.

Results in the output file can also be accessed or exported using **Data
Tools** from the main menu of CellProfiler. The pipeline with its
settings can be be loaded from an output file using *File > Load
Pipeline…*

The output file will be saved in the Default Output Folder unless you
type a full path and file name into the file name box. The path must not
have spaces or characters disallowed by your computer’s platform.

If the output filename ends in *OUT.mat* (the typical text appended to
an output filename), CellProfiler will prevent you from overwriting this
file on a subsequent run by generating a new file name and asking if you
want to use it instead. You can override this behavior by checking the
*Allow overwrite?* box to the right.

For analysis runs that generate a large number of measurements, you may
notice that even though the analysis completes, CellProfiler continues
to use an inordinate amount of your CPU and RAM. This is because the
output file is written after the analysis is completed and can take a
very long time for a lot of measurements. If you do not need this file,
select "*Do not write measurements*" from
the “Measurements file format” drop-down box.

.. _MATLAB: http://www.mathworks.com/products/matlab/
.. _here: http://github.com/CellProfiler/CellProfiler/wiki/Module-structure-and-data-storage-retrieval#HDF5
