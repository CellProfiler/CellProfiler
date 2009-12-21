import cStringIO
import numpy as np
import time
import wx
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

class ProgressFrame(wx.Frame):

    def __init__(self, *args, **kwds):
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)

        self.start_time = time.time()
        self.current_module = None
        self.current_module_start_time = self.start_time
        self.time_per_module = None

        # GUI stuff
        self.BackgroundColour = cellprofiler.preferences.get_background_color()
        self.tbicon = wx.TaskBarIcon()
        self.tbicon.SetIcon(get_icon(), "CellProfiler2.0")
        self.SetTitle("CellProfiler (svn %d)"%(get_revision.version))
        self.SetSize((640, 480))
        self.panel = wx.Panel(self, wx.ID_ANY)
        sizer = wx.BoxSizer(wx.VERTICAL)
        times_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.elapsed_control = wx.StaticText(self.panel, -1, 
                                             label=self.elapsed_label(), 
                                             style=wx.ALIGN_LEFT | wx.ST_NO_AUTORESIZE)
        self.remaining_control = wx.StaticText(self.panel, -1, label="Remaining: 2 min", style=wx.ALIGN_RIGHT | wx.ST_NO_AUTORESIZE)
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
        def get_bitmap(name):
            return wx.BitmapFromImage(wx.ImageFromStream(cStringIO.StringIO(name)))
        buttons_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.play_pause_button = wx.BitmapButton(self.panel, -1, bitmap=get_bitmap(cellprofiler.icons.pause))
        buttons_sizer.Add(self.play_pause_button, 0, wx.ALL, 5)
        self.stop_button = wx.BitmapButton(self.panel, -1, bitmap=get_bitmap(cellprofiler.icons.stop))
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
        elapsed = time.time() - self.start_time
        hours = elapsed // (60 * 60)
        rest = elapsed % (60 * 60)
        minutes = rest // 60
        rest = rest % 60
        seconds = rest
        s = "%d h "%(hours,) if hours > 0 else ""
        s += "%d min "%(minutes,) if hours > 0 or minutes > 0 else ""
        s += "%d s"%(seconds,)
        return "Elapsed: " + s

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
            time_spent = time.time() - self.current_module_start_time
            self.time_per_module[module.module_num - 1] += time_spent
            # Update projection.
            projection = 0.
            for i in range(module.module_num):
                average = self.time_per_module[i] / (image_set_index + 1)
                projection += average * (num_image_sets - image_set_index)
            for i in range(module.module_num, num_modules):
                average = self.time_per_module[i] / image_set_index
                projection += average * (num_image_sets - image_set_index + 1)
        self.current_module = module
        if module:
            self.current_module_start_time = time.time()
        self.current_module_control.SetLabel(module_label(module))
        self.image_set_control.SetLabel(image_set_label(image_set_index + 1, num_image_sets))


if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = ProgressFrame(None).Show()
    app.MainLoop()
    
