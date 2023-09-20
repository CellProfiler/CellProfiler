Using Plugins
=============

Occasionally certain modules aren't installed by default into CellProfiler;
sometimes they are created just for specific projects but our team isn't sure
they are useful for the general public, some are broadly useful but are missing
documentation, some only work for specific versions of CellProfiler, or some 
may introduce extra libraries or other dependencies that for whatever reason 
we're unable or unwilling to include in the overall program.  These modules are
called *Plugins*, and you can find them in their own GitHub `repository`_.  They
are often experimental and may be less likely to work, but you may find a use 
for one or more of them!

Note that if a plugin requires additional libraries which aren't packaged with
CellProfiler, you'll need to build CellProfiler from source rather than using
a pre-packaged version. Installation instructions for your platform can be found
on the GitHub `wiki`_.

You may download these modules individually by clicking on a module's name,  
hitting the "Raw" button on GitHub, then using your browser's Save function.  
You can also download the whole repository of pipelines by cloning the whole 
repository from GitHub using Git, the GitHub app, or by downloading a ZIP file
with all the modules using the "Clone or download" green button on the
repository's landing page.

CellProfiler will check for plugins in its plugins directory, which you can set
from the *File > Preferences* menu.  Once you've obtained the module(s) you're 
interested in, either move them to the plugins directory or set the plugins 
directory to the folder containing your new module(s).  Once you've done so, 
simply restart CellProfiler; if the module loads correctly then you should be 
able to see it in the list of modules and add it to your pipeline. 
If it does not load correctly, we encourage you to please check the log (see 
*Help > Other Features > Configuring Logging* ) for more information then check for known issues 
and/or notify us on the `issues`_ page or the CellProfiler `forum`_.

Additionally, if you write your own module based on our `recommendations`_, you
can add it to your own plugins directory.  If you think it fills an unmet need 
in the CellProfiler code, feel free to contribute it to the CellProfiler-plugins
repository!

.. _repository: http://github.com/CellProfiler/CellProfiler-plugins
.. _issues: http://github.com/CellProfiler/CellProfiler-plugins/issues
.. _forum: https://forum.image.sc/tag/cellprofiler
.. _wiki: https://github.com/CellProfiler/CellProfiler/wiki
.. _recommendations: http://github.com/CellProfiler/CellProfiler/wiki/Orientation-to-CellProfiler-code
