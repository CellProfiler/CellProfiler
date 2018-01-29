Loading Image Stacks and Movies
===============================

In this context, the term *image sequence* is used to refer to a
collection of images from a time-lapse assay (movie), a
three-dimensional (3-D) Z-stack assay, or both. This section will teach
you how to load these collections in order to properly represent your
data for processing.

Sequences of individual files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some microscopes, the simplest method of capturing image sequences
is to simply acquire them as a series of individual image files, where
each image file represents a single timepoint and/or Z-slice. Typically,
the image filename includes the timepoint or Z-slice, such that the
alphabetical image listing corresponds to the proper sequence, e.g.,
*img000.png*, *img001.png*, *img002.png*, etc.

It is also not uncommon to store the movie such that one movie’s worth
of files is stored in a single folder.

*Example:* You have a time-lapse movie of individual files set up as
follows:

-  Three folders, one for each image channel, named *DNA*, *actin* and
   *phase*.
-  In each folder, the files are named as follows:

   -  *DNA*: calibrate2-P01.001.TIF, calibrate2-P01.002.TIF,…,
      calibrate2-P01.287.TIF
   -  *actin*: calibrated-P01.001.TIF, calibrated-P01.002.TIF,…,
      calibrated-P01.287.TIF
   -  *phase*: phase-P01.001.TIF, phase-P01.002.TIF,…, phase-P01.287.TIF

   where the file names are in the format
   *<Stain>-<Well>.<Timepoint>.TIF*.
-  There are 287 timepoints per movie, and a movie of the 3 channels
   above is acquired from each well in a multi-well plate.

In this case, the procedure to set up the input modules to handle these
files is as follows:

-  In the **Images** module, drag-and-drop your folders of images into the
   File list panel. If necessary, set your rules accordingly in order to
   filter out any files that are not part of a movie sequence. By default,
   only image files with `Bio-Formats`_ extensions are included, which
   includes a wide range of image formats and covers most situations.

   In the above example, you would drag-and-drop the *DNA*, *actin* and
   *phase* folders into the File list panel.

-  In the **Metadata** module, check the box to enable metadata
   extraction. The key step here is to obtain the metadata tags
   necessary to do two things:

   -  Distinguish the movies from each other. This information is
      typically encapsulated in the filename and/or the folder name.
   -  For each movie, distinguish the timepoints from each other and
      ensure their proper ordering. This information is usually
      contained in the filename.

   To accomplish this, do the following:

   -  Select “Extract from file/folder names” or “Import from file” as
      the metadata extraction method. You will use these to extract the
      movie and timepoint tags from the images.
   -  Use “Extract from file/folder names” to create a regular expression to
      extract the metadata from the filename and/or path name.
   -  Or, use “Import from file” if you have a comma-delimited
      file (CSV) of the necessary metadata columns (including the movie
      and timepoint tags) for each image. Note that microscopes rarely
      produce such a file, but it might be worthwhile to write scripts
      to create them if you do this frequently.

   If there are multiple channels for each movie, this step may need to
   be performed for each channel.

   In this example, you could do the following:

   -  Select “Extract from file/folder names” as the method, “From file name”
      as the source, and
      ``.*-(?P<Well>[A-P][0-9]{{2}})\.(?P<Timepoint>[0-9]{{3}})`` as the
      regular expression. This step will extract the well ID and
      timepoint from each filename.
   -  Click the “Add” button to add another extraction method.
   -  In the new group of extraction settings, select
      “Extract from file/folder names” as the method, “From folder name” as the
      source, and ``.*[\\/](?P<Stain>.*)[\\/].*$`` as the regular
      expression. This step will extract the stain name from each folder
      name.
   -  Click the “Update” button below the divider and check the output
      in the table to confirm that the proper metadata values are being
      collected from each image.
      
   Note that there are many online tools available to assist with the
   creation of regular expressions that match the patterns within image
   filenames. CellProfiler uses the `Python regular expression format`_.

-  In the **NamesAndTypes** module, assign the channel(s) to a name of
   your choice. If there are multiple channels, you will need to do this
   for each channel. The names will be used throughout the pipeline to
   reference the images imported into CellProfiler, so choose names that
   are descriptive. For example, the name “DAPI” is more descriptive
   than “channel\_1” or “blue”, because it conveys that the contents of
   the image is stained DNA.

   For this example, you could do the following:

   -  Select “Assign images matching rules”.
   -  Make a new rule
      ``[Metadata][Does][Have Stain matching][actin]`` and
      name it *OrigFluor*.
   -  Click the “Add” button to define another image with a rule.
   -  Make a new rule
      ``[Metadata][Does][Have Stain matching][DNA]`` and
      name it *OrigFluo2*.
   -  Click the “Add” button to define another image with a rule.
   -  Make a new rule
      ``[Metadata][Does][Have Stain matching][phase]`` and
      name it *OrigPhase*.
   -  In the “Image set matching method” setting, select “Metadata”.
   -  Select “Well” for the *OrigFluor*, *OrigFluo2*, and *OrigPhase*
      channels.
   -  Click the |icon_add| button to the right to add another row, and
      select “Timepoint” for each channel.
   -  Click the “Update” button below the divider to view the resulting
      table and confirm that the proper files are listed and matched
      across the channels. The corresponding well and frame for each
      channel should now be matched to each other.

-  In the **Groups** module, enable image grouping. Select the metadata
   that defines a distinct movie of data, which could be a well or site or
   x-y location or folder name. Select multiple metadata to refine the set
   of images.

   For the example above, do the following:

   -  Select “Well” as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each well is defined as a
      group, each with 287 frames’ worth of images.

   Without this step, CellProfiler would not know where one movie ends
   and the next one begins, and would process the images in all movies
   together as if they were a single movie. This would result in, for
   example, the TrackObjects module attempting to track cells from the
   end of one movie to the start of the next movie.

If your images represent a 3D image, you can follow the above example to
process your data slice by slice; in other words CellProfiler
will analyze each Z-slice individually and sequentially. However, it is important to note that CellProfiler
will analyze an image stack as a whole volume
(3D image) if "Process as 3D" is selected in **NamesAndTypes**.

Note that CellProfiler only supports single-channel .TIF stacks. More
complex multipage .TIF stacks will need to be “unpacked” before
importing them into CellProfiler. Splitting image channels and
converting image sets into .TIF stacks can be done using another
software application, like FIJI.

Basic image sequences consisting of a single file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Another common means of storing time-lapse or Z-stack data is as a
single file containing frames. Examples of this approach include image
formats such as:

-  Multi-frame or multipage TIF
-  Metamorph stack: STK
-  Evotec/PerkinElmer Opera Flex
-  Zeiss ZVI, LSM
-  Standard movie formats: AVI, Quicktime MOV, etc

CellProfiler uses the Bio-Formats library for reading various image
formats. For more details on supported files, see this `webpage`_. In
general, we recommend saving stacks and movies in .TIF format.

*Example:* You have several image stacks representing 3D structures in
the following format:

-  The stacks are saved in .TIF format.
-  Each stack is a single-channel grayscale image.
-  Your files have names like IMG01\_CH01.TIF, IMG01\_CH02.TIF, …
   IMG01\_CH04.TIF and IMG02\_CH01.TIF, IMG02\_CH02.TIF, etc, where
   IMG01\_CH01.TIF designates channel 1 from image 1, IMG01\_CH02.TIF
   designates channel 2 from image 1, and IMG02\_CH01.TIF designates
   channel 1 from image 2.

Note that the images, such as IMG01\_CH01.TIF, must be a multipage TIF
for a single channel. For example, if 30 Z-slices are acquired during
imaging, then the TIF image will be a 30 slice stack for each channel.
Individual images cannot be grouped together by the **Groups** module
and then processed as a 3D volume.

You would like to process each stack as a single image, not as a series
of 2D images. In this case, the procedure to set up the input modules to
handle these files is as follows:

-  In the **Images** module, drag-and-drop your folders of images into
   the File list panel. If necessary, set your rules accordingly in
   order to filter out any files that are not images to be processed.
   In the above example, you would drag-and-drop the .TIF files into the
   File list panel.
-  In the **NamesAndTypes** module, select “Yes” for “Process as 3D”. You
   should also provide the relative X, Y, and Z pixel sizes of your
   images. X and Y will be determined by the camera and objective you
   used to capture your images. Your Z size represents the spacing of
   your Z-series. In most cases, the X and Y pixel size will be the
   same. You can divide the Z size by X or Y to get a relative value,
   with X = Y = 1. CellProfiler will use this information to correctly
   compute filter sizes and shape features, for example.
   Additionally assign each channel to a name of your choice. You will
   need to do this for each channel. For this example, you could do the
   following:

   -  Select “Assign images matching rules”.
   -  Make a new rule ``[File][Does][Contain][CH01]``
   -  Provide a descriptive name for the channel, e.g., *DAPI*.
   -  Click the “Add another image” button to define a second image with
      a set of rules.
   -  Make a new rule ``[File][Does][Contain][CH02]``
   -  Provide a descriptive name for the channel *GFP*.
   -  Click the “Update” button below the divider to confirm that the
      proper images are listed and matched across the channels. All file
      names ending in CH01.TIF should be matched together.

*Example:* You have two image stacks in the following format:

-  The stacks are Opera’s FLEX format.
-  Each FLEX file contains 8 fields of view, with 3 channels at each
   site (DAPI, GFP, Texas Red).
-  Each channel is in grayscale format.

In this case, the procedure to set up the input modules to handle these
files is as follows:

-  In the **Images** module, drag-and-drop your folders of images into
   the File list panel. If necessary, set your rules accordingly in
   order to filter out any files that are not images to be processed.
   In the above example, you would drag-and-drop the FLEX files into the
   File list panel.
-  In the **Metadata** module, enable metadata extraction in order to
   obtain metadata from these files. The key step here is to obtain the
   necessary metadata tags to do two things:

   -  Distinguish the stacks from each other. This information is
      contained as the file itself, that is, each file represents a
      different stack.
   -  For each stack, distinguish the frames from each other. This
      information is usually contained in the image’s internal metadata,
      in contrast to the image sequence described above.

   To accomplish this, do the following:

   -  Select “Extract from image file headers” as the metadata extraction
      method. In this case, CellProfiler will extract the requisite
      information from the metadata stored in the image headers.
   -  Click the “Update metadata” button. A progress bar will appear
      showing the time elapsed; depending on the number of files
      present, this step may take a while to complete.
   -  Click the “Update” button below the divider.
   -  The resulting table should show the various metadata contained in
      the file. In this case, the relevant information is contained in
      the *C* and *Series* columns. In the figure shown, the *C* column
      shows three unique values for the channels represented, numbered
      from 0 to 2. The *Series* column shows 8 values for the slices
      collected in each stack, numbered from 0 to 7, followed by the
      slices for other stacks.

-  In the **NamesAndTypes** module, assign the channel to a name of your
   choice. If there are multiple channels, you will need to do this for
   each channel. For this example, you could do the following:

   -  Select “Assign images matching rules”.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][0]``
   -  Click the |icon_add| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``. This combination tells
      CellProfiler not to treat the image as a single file, but rather
      as a series of frames.
   -  Name the image *DAPI*.
   -  Click the “Add another image” button to define a second image with
      a set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][1]``
   -  Click the |icon_add| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *GFP*.
   -  Click the “Add another image” button to define a third image with
      a set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][2]``
   -  Click the |icon_add| button to the right of the rule to add another
      set of rules underneath.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *TxRed*.
   -  In the “Image set matching method” setting, select “Metadata”.
   -  Select “FileLocation” for the DAPI, GFP and TxRed channels. The
      FileLocation metadata tag identifies the individual stack, and
      selecting this parameter ensures that the channels are first
      matched within each stack, rather than across stacks.
   -  Click the |icon_add|  button to the right to add another row, and
      select *Series* for each channel.
   -  Click the “Update” button below the divider to confirm that the
      proper image slices are listed and matched across the channels.
      The corresponding *FileLocation* and *Series* for each channel
      should now be matched to each other.

-  In the **Groups** module, select the metadata that defines a distinct
   image stack. For the example above, do the following:

   -  Select “FileLocation” as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each of the two image
      stacks are defined as a group, each with 8 slices’ worth of
      images.

   Without this step, CellProfiler would not know where one stack ends
   and the next one begins, and would process the slices in all stacks
   together as if they were constituents of only one stack.

*Example:* You have four Z-stacks in the following format:

-  The stacks are in Zeiss’ CZI format.
-  Each stack consists of a number of slices with 4 channels (DAPI, GFP,
   Texas Red and Cy3) at each slice.
-  One stack has 9 slices, two stacks have 7 slices and the fourth has
   12 slices. Even though the stacks were collected with differing
   numbers of slices, the pipeline to be constructed is intended to
   analyze all stacks in the same manner.
-  Each slice is in grayscale format.

In this case, the procedure to set up the input modules to handle these
this file is as follows *(note that these Z-stacks will not be processed
as a 3D volume)*:

-  In the **Images** module, drag-and-drop your folders of images into
   the File list panel. If necessary, set your rules accordingly in
   order to filter out any files that are not images to be processed.
   In the above example, you would drag-and-drop the CZI files into the
   File list panel. In this case, the default “Images only” filter is
   sufficient to capture the necessary files.
-  In the **Metadata** module, enable metadata extraction in order to
   obtain metadata from these files. The key step here is to obtain the
   metadata tags necessary to do two things:

   -  Distinguish the stacks from each other. This information is
      contained as the file itself, that is, each file represents a
      different stack.
   -  For each stack, distinguish the z-planes from each other, ensuring
      proper ordering. This information is usually contained in the
      image file’s internal metadata.

   To accomplish this, do the following:

   -  Select “Extract from image file headers” as the metadata extraction
      method. In this case, CellProfiler will extract the requisite
      information from the metadata stored in the image headers.
   -  Click the “Update metadata” button. A progress bar will appear
      showing the time elapsed; depending on the number of files
      present, this step may take a while.
   -  Click the “Update” button below the divider.
   -  The resulting table should show the various metadata contained in
      the file. In this case, the relevant information is contained in
      the C and Z columns. The *C* column shows four unique values for
      the channels represented, numbered from 0 to 3. The *Z* column
      shows nine values for the slices represented from the first stack,
      numbered from 0 to 8.
   -  Of note in this case, for each file there is a single row
      summarizing this information. The *sizeC* column reports a value
      of 4 and *sizeZ* column shows a value of 9. You may need to scroll
      down the table to see this summary for the other stacks.

-  In the **NamesAndTypes** module, assign the channel(s) to a name of
   your choice. If there are multiple channels, you will need to do this
   for each channel.

   For the above example, you could do the following:

   -  Select “Assign images matching rules”.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][0]``
   -  Click the |icon_add| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the image *DAPI*.
   -  Click the “Add another image” button to define a second image with
      a set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][1]``
   -  Click the |icon_add| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the second image *GFP*.
   -  Click the “Add another image” button to define a third image with
      a set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][2]``.
   -  Click the |icon_add| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the third image *TxRed*.
   -  Click the “Add another image” button to define a fourth image with
      set of rules.
   -  Make a new rule
      ``[Metadata][Does][Have C matching][3]``.
   -  Click the |icon_add| button to the right of the rule to add another
      set of rule options.
   -  Add the rule ``[Image][Is][Stack frame]``.
   -  Name the fourth image *Cy3*.
   -  In the “Image set matching method” setting, select “Metadata”.
   -  Select “FileLocation” for the *DAPI*,\ *GFP*,\ *TxRed*, and
      *Cy3*\ channels. The *FileLocation* identifies the individual
      stack, and selecting this parameter insures that the channels are
      matched within each stack, rather than across stacks.
   -  Click the |icon_add| button to the right to add another row, and
      select “Z” for each channel.
   -  Click “Update table” to confirm the channel matching. The
      corresponding *FileLocation* and *Z* for each channel should be
      matched to each other.

-  In the **Groups** module, select the metadata that defines a distinct
   image stack. For the example above, do the following:

   -  Select “FileLocation” as the metadata category.
   -  The tables below this setting will update themselves, and you
      should be able to visually confirm that each of the four image
      stacks are defined as a group, with 9, 7, 7 and 12 slices’ worth
      of images.

   Without this step, CellProfiler would not know where one stack ends
   and the next one begins, and would process the slices in all stacks
   together as if they were constituents of only one stack.

.. _Bio-Formats: http://docs.openmicroscopy.org/bio-formats/5.7.1/supported-formats.html
.. _Python regular expression format: http://docs.python.org/2.7/howto/regex.html
.. _webpage: http://docs.openmicroscopy.org/bio-formats/5.6.0/supported-formats.html

.. |icon_add| image:: ../images/module_add.png
