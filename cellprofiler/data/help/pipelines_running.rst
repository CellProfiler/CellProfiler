Running Your Pipeline
=====================

Once you have tested your pipeline using Test mode and you are satisfied
with the module settings, you are ready to run the pipeline on your
entire set of images. To do this:

-  Exit Test mode by clicking the “Exit Test Mode” button or selecting
   *Test > Exit Test Mode*.
-  Click the "|HelpContent_RunningPipeline_image0| Analyze Images" button and begin processing your
   data sets.

During the analysis run, the progress will appear in the status bar at
the bottom of CellProfiler. It will show you the total number of image
sets, the number of image sets completed, the time elapsed and the
approximate time remaining in the run.

If you need to pause analysis, click the "|HelpContent_RunningPipeline_image1| Pause" button, then
click the “Resume” button to continue. If you want to terminate
analysis, click the "|HelpContent_RunningPipeline_image2| Stop Analysis" button.

If your computer has multiple processors, CellProfiler will take
advantage of them by starting multiple copies of itself to process the
image sets in parallel. You can set the number of *workers* (i.e., copies
of CellProfiler activated) under *File > Preferences…*

.. |HelpContent_RunningPipeline_image0| image:: ../images/IMG_ANALYZE_16.png
.. |HelpContent_RunningPipeline_image1| image:: ../images/IMG_PAUSE.png
.. |HelpContent_RunningPipeline_image2| image:: ../images/stop.png
