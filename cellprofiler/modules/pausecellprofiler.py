'''<b>Pause CellProfiler</b> pauses CellProfiler during analysis 
<hr>

This module allows you to pause CellProfiler's processing at the point where the module
resides in the pipeline, which can be helpful if you want to examine the results before proceeding. 
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
import cellprofiler.preferences as cpprefs
try:
    import wx
except:
    pass

class PauseCellProfiler(cpm.CPModule):
    module_name = "PauseCellProfiler"
    variable_revision_number = 1
    category = "Other"
    
    def create_settings(self):
        """Create your settings by subclassing this function
        
        create_settings is called at the end of initialization. You should
        name your module in this routine:
        
            # Set the name that will appear in the "AddModules" window
            self.module_name = "My module"
        
        You should also create the setting variables for your module:
            # Ask the user for the input image
            self.image_name = cellprofiler.settings.ImageNameSubscriber(...)
            # Ask the user for the name of the output image
            self.output_image = cellprofiler.settings.ImageNameProvider(...)
            # Ask the user for a parameter
            self.smoothing_size = cellprofiler.settings.Float(...)
        """
        self.action = cps.Choice("Pause here, skip subsequent modules or continue without prompting?",
                                [cpw.DISPOSITION_PAUSE, cpw.DISPOSITION_SKIP, cpw.DISPOSITION_CONTINUE],
                                doc = """
                                There are three options:
                                <ul>
                                <li><i>%s</i> will pause CellProfiler
                                at this module's position in the pipeline. The pipeline will stop 
                                and a window with a <i>Resume</i>
                                button will appear.
                                The pipeline will continue when you hit the <i>Resume</i> button or will
                                stop if you hit the <i>Stop analysis</i> button on the main window.</li>
                                <li><i>%s</i> will pause as described above, but if you choose to resume, 
                                CellProfiler will skip all modules following the <b>PauseCellProfiler</b> 
                                module and will advance to begin applying the first module in the pipeline
                                to the next image cycle.</li>
                                <li><i>%s</i> will continue pipeline execution without stopping. This 
                                enables you to temporarily run the full pipeline without the inconvenience 
                                of removing the <b>PauseCellProfiler</b> module from the pipeline.</li>
                                </ul>""" %
                                (cpw.DISPOSITION_PAUSE, cpw.DISPOSITION_SKIP, cpw.DISPOSITION_CONTINUE))
    
    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline
        
        These are the settings (from cellprofiler.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [self.action]
        
    def run(self, workspace):
        """Run the module
        
        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.
        """
        if workspace.pipeline.in_batch_mode():
            return
        
        if self.action != cpw.DISPOSITION_CONTINUE:
            #
            # Tell the pipeline to pause after we return
            #
            workspace.disposition = cpw.DISPOSITION_PAUSE
            #
            # Make a frame to hold the resume button
            #
            frame = wx.Frame(None,title="Pause CellProfiler")
            frame.BackgroundColour = cpprefs.get_background_color()
            #
            # Register to hear about the run ending so we can clean up
            #
            def on_pipeline_event(caller, event):
                if isinstance(event, cpp.EndRunEvent):
                    frame.Close()
            workspace.pipeline.add_listener(on_pipeline_event)
            #
            # Make sure we stop listening when the window closes
            #
            def on_close(event):
                if workspace.disposition == cpw.DISPOSITION_PAUSE:
                    workspace.disposition = self.action.value
                workspace.pipeline.remove_listener(on_pipeline_event)
                frame.Destroy()
            frame.Bind(wx.EVT_CLOSE, on_close)
            #
            # Make the UI:
            super_sizer = wx.BoxSizer(wx.VERTICAL)
            frame.SetSizer(super_sizer)
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            super_sizer.Add(sizer, 0, wx.EXPAND)
            sizer.Add(wx.StaticBitmap(frame, -1, 
                                      wx.ArtProvider.GetBitmap(wx.ART_INFORMATION,
                                                               wx.ART_MESSAGE_BOX)),
                      0, wx.EXPAND | wx.ALL, 5)
            sizer.Add(wx.StaticText(frame,-1, """Press "Resume" to continue processing.\nPress "Cancel" to stop."""),
                      0, wx.EXPAND | wx.ALL, 5)
            super_sizer.Add(wx.StaticLine(frame), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)
            sizer = wx.StdDialogButtonSizer()
            super_sizer.Add(sizer, 0, wx.EXPAND)
            resume_id = 100
            cancel_id = 101
            resume_button = wx.Button(frame, resume_id, "Resume")
            sizer.AddButton(resume_button)
            sizer.SetAffirmativeButton(resume_button)
            cancel_button = wx.Button(frame, cancel_id, "Cancel")
            sizer.AddButton(cancel_button)
            sizer.SetNegativeButton(cancel_button)
            sizer.Realize()
            frame.Fit()
            
            def on_resume(event):
                if self.action == cpw.DISPOSITION_PAUSE:
                    workspace.disposition = cpw.DISPOSITION_CONTINUE
                else:
                    workspace.disposition = self.action.value
                frame.Close()
            
            def on_cancel(event):
                workspace.disposition = cpw.DISPOSITION_CANCEL
                frame.Close()
            
            frame.Bind(wx.EVT_BUTTON, on_resume, id=resume_id)
            frame.Bind(wx.EVT_BUTTON, on_cancel, id=cancel_id)
            frame.Show()
    
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        '''Adjust setting values if they came from a previous revision
        
        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        from_matlab - True if the settings came from a Matlab pipeline, False
                      if the settings are from a CellProfiler 2.0 pipeline.
        
        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        '''
        if from_matlab and variable_revision_number == 1:
            setting_values = [setting_values[0], 'Pause']
            variable_revision_number = 2
        if from_matlab and variable_revision_number == 2:
            setting_values = [setting_values[1]]
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab
        
