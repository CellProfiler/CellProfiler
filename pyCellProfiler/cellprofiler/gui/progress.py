import cStringIO
import numpy as np
import time
import wx
import sys
import cellprofiler
import cellprofiler.icons
import cellprofiler.utilities.get_revision as get_revision
from cellprofiler.gui import get_icon, get_cp_bitmap

def module_label(module):
    if module:
        return "Current module: " + module.module_name
    else:
        return "Current module:"

def image_set_label(image_set_index, num_image_sets):
    if image_set_index:
        return "Image set: %(image_set_index)d of %(num_image_sets)d"%locals()
    else:
        return "Image set:"

if sys.platform.startswith("win"):
    # :-p
    PROGRESS_FRAME_STYLE = wx.DEFAULT_FRAME_STYLE | wx.STAY_ON_TOP
else:
    # :-(
    PROGRESS_FRAME_STYLE = wx.DEFAULT_FRAME_STYLE

class ProgressFrame(wx.Frame):

    def __init__(self, *args, **kwds):
        kwds["style"] = PROGRESS_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)

        self.start_time = time.time()
        self.current_module = None
        self.current_module_start_time = self.start_time
        self.time_per_module = None
        self.all_times_per_module = None
        self.elapsed_pause_time = 0
        self.pause_start_time = None
        self.paused = False

        # GUI stuff
        self.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.tbicon = wx.TaskBarIcon()
        self.tbicon.SetIcon(get_icon(), "CellProfiler2.0")
        self.SetTitle("CellProfiler (v.%d)"%(get_revision.version))
        self.SetSize((640, 480))
        self.panel = wx.Panel(self, wx.ID_ANY)
        sizer = wx.BoxSizer(wx.VERTICAL)
        times_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.elapsed_control = wx.StaticText(self.panel, -1, 
                                             label=self.elapsed_label(), 
                                             style=wx.ALIGN_LEFT)
        self.remaining_control = wx.StaticText(self.panel, -1, 
                                               label="Remaining: unknown - calculating", 
                                               style=wx.ALIGN_RIGHT)
        times_sizer.Add(self.elapsed_control, 1, wx.ALIGN_LEFT | wx.ALL, 5)
        times_sizer.Add(self.remaining_control, 1, wx.ALIGN_RIGHT | wx.ALL, 5)
        sizer.Add(times_sizer, 0, wx.EXPAND)
        self.gauge = wx.Gauge(self.panel, -1, style=wx.GA_HORIZONTAL)
        self.gauge.SetValue(30)
        sizer.Add(self.gauge, 0, wx.ALL | wx.EXPAND, 5)
        self.image_set_control = wx.StaticText(self.panel, -1, label=image_set_label(None, None))
        sizer.Add(self.image_set_control, 0, wx.LEFT | wx.RIGHT, 5)
        self.current_module_control = wx.StaticText(self.panel, -1, label=module_label(None))
        sizer.Add(self.current_module_control, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.play_pause_button = wx.BitmapButton(self.panel, -1, 
                                                 bitmap=wx.BitmapFromImage(cellprofiler.icons.pause))
        buttons_sizer.Add(self.play_pause_button, 0, wx.ALL, 5)
        self.stop_button = wx.BitmapButton(self.panel, -1, bitmap=wx.BitmapFromImage(cellprofiler.icons.stop))
        buttons_sizer.Add(self.stop_button, 0, wx.ALL, 5)
        sizer.Add(buttons_sizer, 0, wx.CENTER)
        self.panel.SetSizer(sizer)
        sizer.Fit(self)

        # Timer that updates elapsed
        timer_id = wx.NewId()
        self.timer = wx.Timer(self.panel, timer_id)
        self.timer.Start(100)
        wx.EVT_TIMER(self.panel, timer_id, self.on_timer)

    def elapsed_label(self):
        elapsed = self.elapsed_time()
        hours = elapsed // (60 * 60)
        rest = elapsed % (60 * 60)
        minutes = rest // 60
        rest = rest % 60
        seconds = rest
        s = "%d h "%(hours,) if hours > 0 else ""
        s += "%d min "%(minutes,) if hours > 0 or minutes > 0 else ""
        s += "%d s"%(seconds,)
        return "Elapsed: " + s
    
    def elapsed_time(self):
        '''Return the # of seconds that have elapsed since start
        
        accounts for paused time
        '''
        if self.paused:
            return (self.pause_start_time - self.start_time - 
                    self.elapsed_pause_time)
        else:
            return time.time() - self.start_time - self.elapsed_pause_time
        
    def pause(self):
        self.play_pause_button.SetBitmapLabel(
            wx.BitmapFromImage(cellprofiler.icons.play))
        self.pause_start_time = time.time()
        self.paused = True
        
    def play(self):
        self.play_pause_button.SetBitmapLabel(
            wx.BitmapFromImage(cellprofiler.icons.pause))
        self.elapsed_pause_time += time.time() - self.pause_start_time
        self.paused = False
        
    def on_timer(self, event):
        self.elapsed_control.SetLabel(self.elapsed_label())
        self.timer.Start(100)

    def OnClose(self, event):
        self.tbicon.Destroy()
        self.Destroy()

    def start_module(self, module, num_modules, image_set_index, 
                     num_image_sets):
        self.num_modules = num_modules

        if self.current_module: # and False:  # Disable untested code
            # Record time spent on previous module.
            if self.time_per_module is None:
                self.time_per_module = np.zeros(num_modules)
                self.gauge.SetRange(num_image_sets)
            self.gauge.Value = image_set_index
            self.gauge.Refresh()
            if self.all_times_per_module is None:
                self.all_times_per_module = np.zeros((num_modules, num_image_sets))
            time_spent = time.time() - self.current_module_start_time
            self.time_per_module[module.module_num - 1] += time_spent
            self.all_times_per_module[module.module_num -1, image_set_index] = time_spent
            # Update projection.
            projection = 0.
            for i in range(module.module_num):
                average = self.time_per_module[i] / (image_set_index + 1)
                projection += average * (num_image_sets - image_set_index)
            for i in range(module.module_num, num_modules):
                average = self.time_per_module[i] / image_set_index
                projection += average * (num_image_sets - image_set_index + 1)
            if image_set_index > 0:
                # Estimate amount of time remaining as the median time
                # for each module times the number of image sets remaining
                median_time_per_module = np.median(
                    self.all_times_per_module[:,:image_set_index],1)
                time_per_image_set = np.sum(median_time_per_module)
                time_remaining = int(time_per_image_set * 
                                     (num_image_sets - image_set_index))
                sec_remaining = time_remaining % 60
                min_remaining = (time_remaining / 60) % 60
                hr_remaining = (time_remaining / 60 / 60) % 24
                day_remaining = (time_remaining / 60 / 60 / 24)
                s = ("Remaining:" +
                     ((" %d d"%day_remaining) if day_remaining > 0 else "") +
                     ((" %d h"%hr_remaining) if hr_remaining > 0 else "") +
                     ((" %d m"%min_remaining) if min_remaining > 0 else "") +
                     ((" %d s"%sec_remaining) if sec_remaining > 0 else ""))
                self.remaining_control.Label = s
                self.remaining_control.Refresh()
                
        self.current_module = module
        if module:
            self.current_module_start_time = time.time()
        self.current_module_control.SetLabel(module_label(module))
        self.current_module_control.Refresh()
        self.image_set_control.SetLabel(image_set_label(image_set_index + 1, num_image_sets))
        self.image_set_control.Refresh()


if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = ProgressFrame(None).Show()
    app.MainLoop()
    
