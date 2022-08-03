Using the Test Menu
===================

Before starting an analysis run, you can test the pipeline settings on a
selected image cycle using the *Test* mode option on the main menu. Test
mode allows you to run the pipeline on a selected image, preview the
results and adjust the module settings on the fly.

To enter Test mode once you have built a pipeline, choose *Test > Start
Test Mode* from the menu bar in the main window. At this point, you will
see the following features appear:

-  The buttons available at the bottom of the pipeline panel change.
-  A Pause icon |HelpContent_TestMode_image_pause|  will appear to the left
   of each module.
-  A Step icon |HelpContent_TestMode_image_step| will appear next to the
   module which is due to be executed next. Once executed this icon will
   change to be dimmer  |HelpContent_TestMode_image_stepped|. You can click on
   these icons to jump back and execute these modules again.

You can run your pipeline in Test mode by selecting *Test > Step to Next
Module* or clicking the *Run* or *Step* buttons at the bottom of the
pipeline panel. The pipeline will execute normally, but you will be able
to back up to a previous module or jump to a downstream module, change
module settings to see the results, or execute the pipeline on the image
of your choice. The additional controls allow you to do the following:

-  *Run from module N:* Start or resume execution of the pipeline at any
   time from a selected module. Right-click the module
   and select "Run from module N", where "N" is the module number.
   This menu option is only available from modules which have already been
   run in test mode, or from the current module. Test mode will run until
   it reaches the end of the pipeline or it encounters a pause.
-  *Pause:* Clicking the pause icon will cause the pipeline test run to
   halt execution when that module is reached (the paused module itself
   is not executed). The icon changes from |HelpContent_TestMode_image_pause|
   to |HelpContent_TestMode_image_paused| to indicate that a pause has been
   inserted at that point.
-  *Run:* Execution of the pipeline will be started/resumed until the
   next module pause is reached. When all modules have been executed for
   a given image cycle, execution will stop.
-  *Step:* Execute the next module, as indicated by being underlined and having
   the |HelpContent_TestMode_image_step| icon in front of it.
   If you click once on a module, for example to view or change its settings,
   the active module text (the one you're looking at in the GUI) is **bolded**.
-  *Next Image:* Skip ahead to the next image cycle as determined by the
   image order in the Input modules. The first module in the pipeline will automatically become active (**bolded**) and will run next (underlined).

From the *Test* menu, you can choose additional options:

-  *Start/Exit Test Mode:* Start or Exit *Test* mode. Loading a new pipeline or
   subtracting modules will also automatically exit test mode.
-  *Step to Next Module:* Execute the next module (as indicated by being
   underlined)
-  *Next Image Set:* Step to the next image set in the current image
   group.
-  *Next Image Group:* Step to the next group in the image set. The first
   module in the pipeline will automatically become active (**bolded**) and will run next (underlined).
-  *Random Image Set:* Randomly select and jump to an image set in the
   current image group.
-  *Random Image Group:* Randomly select and jump to an image group.
-  *Choose Image Set:* Choose the image set to jump to. The first module
   in the pipeline will automatically become active (**bolded**) and will run next (underlined).
-  *Choose Image Group:* Choose an image group to jump to. The first module
   in the pipeline will automatically become active (**bolded**) and will run next (underlined).
-  *Reload Modules Source (enabled only if running from source code):*
   This option will reload the module source code, so any changes to the
   code will be reflected immediately.
-  *Break into debugger (enabled only if running from source code):*
   This option will allow you to open a debugger in the terminal window.
-  *Pipeline Testing Help:* Displays help for the test menu.

Note that if movies are being loaded, the individual movie is defined as
a group automatically. Selecting *Choose Image Group* will allow you to
choose the movie file, and *Choose Image Set* will let you choose the
individual movie frame from that file.

Please see the **Groups** module for more details on the proper use of
metadata for grouping.

The *Test* menu also provides a widget inspector. 
See *Help > Other Features > Widget Inspector* for details.

.. |HelpContent_TestMode_image_step| image:: ../images/IMG_ANALYZE_16.png
.. |HelpContent_TestMode_image_stepped| image:: ../images/IMG_ANALYZED.png
.. |HelpContent_TestMode_image_paused| image:: ../images/IMG_PAUSE.png
.. |HelpContent_TestMode_image_pause| image:: ../images/IMG_GO_DIM.png
