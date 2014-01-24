"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import os
import sys
import wx
import wx.html
import content
import webbrowser
import urllib
import urllib2
import cellprofiler.preferences as cpprefs
from cellprofiler.icons import get_builtin_images_path
from cellprofiler.gui.html.content import WELCOME_HELP

MEMORY_SCHEME = "memory:"
WELCOME_SCREEN_FRAME = "WelcomeScreenFrame"
class HtmlClickableWindow(wx.html.HtmlWindow):
    def __init__(self, *args, **kwargs):
        wx.html.HtmlWindow.__init__(self, *args, **kwargs)
        self.HTMLBackgroundColour = cpprefs.get_background_color()

    def load_startup_blurb(self):
        self.OnLinkClicked(wx.html.HtmlLinkInfo('startup_main', ''))

    def OnLinkClicked(self, linkinfo):
        href = linkinfo.Href
        if href.startswith("#"):
            super(HtmlClickableWindow, self).OnLinkClicked(linkinfo)
        elif href.startswith('http://'):
            webbrowser.open(href)
        elif href.startswith('pref:'):
            if 'no_display' in href:
                cpprefs.set_startup_blurb(False)
                # Find the parent frame and, if it's the welcome screen frame,
                # "close" it (= hide it)
                #
                parent = self.Parent
                while parent != None:
                    if parent.Name == WELCOME_SCREEN_FRAME:
                        parent.Close()
                        break
                    parent = parent.Parent
        elif href.startswith('help:'):
            href = linkinfo.Href[7:]
            html_str = WELCOME_HELP[href]
            html_str += '<p>Go <a href="startup_main">back</a> to the welcome screen.</p>'
            self.SetPage(html_str)
            self.BackgroundColour = cpprefs.get_background_color()
        elif href.startswith('load:'):
            pipeline_filename = href[5:]
            try:
                wx.CallAfter(wx.GetApp().frame.pipeline.load, urllib2.urlopen(pipeline_filename))
            except:
                wx.MessageBox(
                    'CellProfiler was unable to load %s' %
                    pipeline_filename, "Error loading pipeline",
                    style = wx.OK | wx.ICON_ERROR)
        elif href.startswith('loadexample:'):
            # Same as "Load", but specific for example pipelines so the user can be directed as to what to do next.
            pipeline_filename = href[12:]
            
            try:
                import cellprofiler.modules.loaddata
                fd = urllib.urlopen(pipeline_filename)
                def fn(fd=fd):
                    pipeline = wx.GetApp().frame.pipeline
                    pipeline.load(fd)
                    for module in pipeline.modules():
                        if isinstance(module, cellprofiler.modules.loaddata.LoadData):
                            # Would prefer to call LoadData's do_reload but not sure how at this point
                            global header_cache
                            header_cache = {}
                            try:
                                module.open_csv()
                            except:
                                pass
                    wx.MessageBox('Now that you have loaded an example pipeline, press the "Analyze images" button to access and process a small image set from the CellProfiler website so you can see how CellProfiler works.', '', wx.ICON_INFORMATION)
                wx.CallAfter(fn)             
            #try:
                #wx.CallAfter(wx.GetApp().frame.pipeline.load, urllib2.urlopen(pipeline_filename))
                #wx.CallAfter(wx.MessageBox,
                             #'Now that you have loaded an example pipeline, press the "Analyze images" button to access and process a small image set from the CellProfiler website so you can see how CellProfiler works.', '', wx.ICON_INFORMATION)
            except:
                wx.MessageBox(
                    'CellProfiler was unable to load %s' %
                    pipeline_filename, "Error loading pipeline",
                    style = wx.OK | wx.ICON_ERROR)
        else:
            newpage = content.find_link(href)
            if newpage is not None:
                self.SetPage(newpage)
                self.BackgroundColour = cpprefs.get_background_color()
            else:
                super(HtmlClickableWindow, self).OnLinkClicked(linkinfo)

    def OnOpeningURL(self, type, url):
        if type == wx.html.HTML_URL_IMAGE:
            if url.startswith(MEMORY_SCHEME):
                path = get_builtin_images_path()
                full_path = os.path.join(path, url[len(MEMORY_SCHEME):])
                if sys.platform.startswith("win"):
                    my_url = full_path
                else:
                    my_url = "file:" + urllib.pathname2url(full_path)
                return my_url
        return wx.html.HTML_OPEN

if __name__ == '__main__':
    app = wx.App(0)
    frame = wx.Frame(None, -1, 'foo', (500, 500))
    htmlwin = HtmlClickableWindow(frame, wx.ID_ANY, style=wx.NO_BORDER)
    frame.Show(True)
    frame.Center()
    app.MainLoop()
