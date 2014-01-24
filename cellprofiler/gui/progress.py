"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import cStringIO
import numpy as np
import time
import wx
import sys
import cellprofiler
from cellprofiler.icons import get_builtin_image
import cellprofiler.utilities.version as version
from cellprofiler.gui import get_cp_icon, get_cp_bitmap

def module_label(module):
    if module:
        return "Current module: " + module.module_name
    else:
        return "Current module:"

def image_set_label(image_set_index, num_image_sets):
    if image_set_index:
        return "Image cycle: %(image_set_index)d of %(num_image_sets)d"%locals()
    else:
        return "Image cycle:"

def duration_label(duration):
    dur = int(round(duration))
    hours = dur // (60 * 60)
    rest = dur % (60 * 60)
    minutes = rest // 60
    rest = rest % 60
    seconds = rest
    s = "%d h "%(hours,) if hours > 0 else ""
    s += "%d min "%(minutes,) if hours > 0 or minutes > 0 else ""
    s += "%d s"%(seconds,)
    return s


class ProgressFrame(wx.Frame):

    def __init__(self, *args, **kwds):
        if sys.platform.startswith("win"):
            kwds["style"] = wx.DEFAULT_FRAME_STYLE | wx.STAY_ON_TOP
        wx.Frame.__init__(self, *args, **kwds)
        self.Show()
        if wx.Platform == '__WXMAC__' and hasattr(self, 'MacGetTopLevelWindowRef'):
            try:
                from AppKit import NSWindow, NSApp, NSFloatingWindowLevel
                window_ref = self.MacGetTopLevelWindowRef()
                nsw = NSWindow.alloc().initWithWindowRef_(window_ref)
                nsw.setLevel_(NSFloatingWindowLevel)
            except ImportError:
                print "No AppKit module => can't make progress window stay on top."

        self.start_time = time.time()
        self.end_times = None
        self.current_module = None
        self.pause_start_time = None
        self.previous_pauses_duration = 0.

        # GUI stuff
        self.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.tbicon = wx.TaskBarIcon()
        self.tbicon.SetIcon(get_cp_icon(), "CellProfiler2.0")
        self.SetTitle("CellProfiler %s"%(version.title_string))
        self.SetSize((640, 480))
        self.panel = wx.Panel(self, wx.ID_ANY)
        sizer = wx.BoxSizer(wx.VERTICAL)
        times_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.elapsed_control = wx.StaticText(self.panel, -1, 
                                             label=self.elapsed_label(), 
                                             style=wx.ALIGN_LEFT)
        self.remaining_control = wx.StaticText(self.panel, -1, 
                                               label=self.remaining_label(), 
                                               style=wx.ALIGN_RIGHT)
        times_sizer.Add(self.elapsed_control, 1, wx.ALIGN_LEFT | wx.ALL, 5)
        times_sizer.Add(self.remaining_control, 1, wx.ALIGN_RIGHT | wx.ALL, 5)
        sizer.Add(times_sizer, 0, wx.EXPAND)
        self.gauge = wx.Gauge(self.panel, -1, style=wx.GA_HORIZONTAL)
        self.gauge.SetValue(0)
        self.gauge.SetRange(100)
        sizer.Add(self.gauge, 0, wx.ALL | wx.EXPAND, 5)
        self.image_set_control = wx.StaticText(self.panel, -1, label=image_set_label(None, None))
        sizer.Add(self.image_set_control, 0, wx.LEFT | wx.RIGHT, 5)
        self.current_module_control = wx.StaticText(self.panel, -1, label=module_label(None))
        sizer.Add(self.current_module_control, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.play_pause_button = wx.BitmapButton(self.panel, -1, 
                                                 bitmap=wx.BitmapFromImage(get_builtin_image('pause')))
        self.play_pause_button.SetToolTipString("Pause")
        buttons_sizer.Add(self.play_pause_button, 0, wx.ALL, 5)
        self.stop_button = wx.BitmapButton(self.panel, -1, bitmap=wx.BitmapFromImage(get_builtin_image('stop')))
        self.stop_button.SetToolTipString("Stop")
        buttons_sizer.Add(self.stop_button, 0, wx.ALL, 5)
        save_bitmap = wx.ArtProvider.GetBitmap(wx.ART_FILE_SAVE,
                                               wx.ART_CMN_DIALOG, 
                                               (16,16))
        self.save_button = wx.BitmapButton(self.panel, -1, bitmap = save_bitmap)
        self.save_button.SetToolTipString("Save measurements")
        buttons_sizer.Add(self.save_button, 0, wx.ALL, 5)
        sizer.Add(buttons_sizer, 0, wx.CENTER)
        self.panel.SetSizer(sizer)
        sizer.Fit(self)

        # Timer that updates elapsed
        timer_id = wx.NewId()
        self.timer = wx.Timer(self.panel, timer_id)
        self.timer.Start(500)
        wx.EVT_TIMER(self.panel, timer_id, self.on_timer)

    def elapsed_label(self):
        return "Elapsed: " + duration_label(self.elapsed_time())
    
    def elapsed_time(self):
        '''Return the number of seconds that have elapsed since start
           as a float.  Pauses are taken into account.
        '''
        return self.adjusted_time() - self.start_time

    def remaining_label(self):
        remaining = self.remaining_time()
        if remaining is None:
            s = "unknown - calculating"
        else:
            s = duration_label(remaining)
        return "Remaining: " + s
        
    def pause(self):
        self.play_pause_button.SetBitmapLabel(
            wx.BitmapFromImage(get_builtin_image('play')))
        self.play_pause_button.SetToolTipString("Resume")
        self.pause_start_time = time.time()
        self.paused = True
        
    def play(self):
        self.play_pause_button.SetBitmapLabel(
            wx.BitmapFromImage(get_builtin_image('pause')))
        self.play_pause_button.SetToolTipString("Pause")
        self.previous_pauses_duration += time.time() - self.pause_start_time
        self.pause_start_time = None
        self.paused = False
        
    def on_timer(self, event):
        self.elapsed_control.SetLabel(self.elapsed_label())
        self.remaining_control.SetLabel(self.remaining_label())
        remaining = self.remaining_time()
        if remaining:
            self.gauge.SetValue(100 * self.elapsed_time() / (self.elapsed_time() + remaining))

    def OnClose(self, event):
        self.timer.Stop()
        self.tbicon.Destroy()
        self.Destroy()

    def adjusted_time(self):
        """Current time minus the duration spent in pauses."""
        pauses_duration = self.previous_pauses_duration
        if self.pause_start_time:
            pauses_duration += time.time() - self.pause_start_time
        return time.time() - pauses_duration

    def start_module(self, module, num_modules, image_set_index, 
                     num_image_sets):
        """
        Update the historical execution times, which are used as the
        bases for projecting the time that remains.  Also update the
        labels that show the current module and image set.  This
        method is called by the pipelinecontroller at the beginning of
        every module execution to update the progress bar.
        """
        self.current_module = module
        self.num_modules = num_modules
        self.image_set_index = image_set_index
        self.num_image_sets = num_image_sets

        if self.end_times is None:
            # One extra element at the beginning for the start time
            self.end_times = np.zeros(1 + num_modules * num_image_sets)
        module_index = module.module_num - 1  # make it zero-based
        index = image_set_index * num_modules + (module_index - 1)
        self.end_times[1 + index] = self.adjusted_time()

        self.current_module_control.SetLabel(module_label(module))
        self.current_module_control.Refresh()
        self.image_set_control.SetLabel(image_set_label(image_set_index + 1, num_image_sets))
        self.image_set_control.Refresh()

    def remaining_time(self):
        """Return our best estimate of the remaining duration, or None
        if we have no bases for guessing."""
        if self.end_times is None:
            return None # We have not started the first module yet
        else:
            module_index = self.current_module.module_num - 1
            index = self.image_set_index * self.num_modules + module_index
            durations = (self.end_times[1:] - self.end_times[:-1]).reshape(self.num_image_sets, self.num_modules)
            per_module_estimates = np.zeros(self.num_modules)
            per_module_estimates[:module_index] = np.median(durations[:self.image_set_index+1,:module_index], 0)
            current_module_so_far = self.adjusted_time() - self.end_times[1 + index - 1]
            if self.image_set_index > 0:
                per_module_estimates[module_index:] = np.median(durations[:self.image_set_index,module_index:], 0)
                per_module_estimates[module_index] = max(per_module_estimates[module_index], current_module_so_far)
            else:
                # Guess that the modules that haven't finished yet are
                # as slow as the slowest one we've seen so far.
                per_module_estimates[module_index] = current_module_so_far
                per_module_estimates[module_index:] = per_module_estimates[:module_index+1].max()
            if False:
                print "current_module_so_far =", current_module_so_far, "; adjusted_time =", self.adjusted_time(), "; end_times =", self.end_times
                print "durations:"
                print durations
                print "per_module_estimates:"
                print per_module_estimates
            per_module_estimates[:module_index] *= self.num_image_sets - self.image_set_index - 1
            per_module_estimates[module_index:] *= self.num_image_sets - self.image_set_index
            per_module_estimates[module_index] -= current_module_so_far
            return per_module_estimates.sum()
        

if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = ProgressFrame(None).Show()
    app.MainLoop()
    
