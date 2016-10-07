# coding=utf-8
from cellprofiler.preferences import get_background_color, set_startup_blurb

import cellprofiler.icons
import content
import os
import sys
import urllib
import urllib2
import webbrowser
import wx
import wx.html

MEMORY_SCHEME = "memory:"
WELCOME_SCREEN_FRAME = "WelcomeScreenFrame"


class HtmlClickableWindow(wx.html.HtmlWindow):
    def __init__(self, *args, **kwargs):
        wx.html.HtmlWindow.__init__(self, *args, **kwargs)

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
                set_startup_blurb(False)
                # Find the parent frame and, if it's the welcome screen frame,
                # "close" it (= hide it)
                #
                parent = self.Parent
                while parent is not None:
                    if parent.Name == WELCOME_SCREEN_FRAME:
                        parent.Close()
                        break
                    parent = parent.Parent
        elif href.startswith('help:'):
            href = linkinfo.Href[7:]
            import cellprofiler.gui.html.content
            html_str = cellprofiler.gui.html.content.WELCOME_HELP[href]
            html_str += '<p>Go <a href="startup_main">back</a> to the welcome screen.</p>'
            self.SetPage(html_str)
        elif href.startswith('load:'):
            pipeline_filename = href[5:]
            try:
                fd = urllib2.urlopen(pipeline_filename)
                if fd.code < 200 or fd.code > 299:
                    wx.MessageBox(
                            "Sorry, the link, \"%s\" is broken, please contact the webmaster" %
                            pipeline_filename,
                            caption="Unable to access pipeline via internet",
                            style=wx.OK | wx.ICON_INFORMATION)
                    return
                wx.CallAfter(wx.GetApp().frame.pipeline.load, fd)
            except:
                wx.MessageBox(
                        'CellProfiler was unable to load %s' %
                        pipeline_filename, "Error loading pipeline",
                        style=wx.OK | wx.ICON_ERROR)
        elif href.startswith('loadexample:'):
            # Same as "Load", but specific for example pipelines so the user can be directed as to what to do next.
            pipeline_filename = href[12:]

            try:
                import cellprofiler.modules.loaddata
                fd = urllib.urlopen(pipeline_filename)
                if fd.code < 200 or fd.code > 299:
                    wx.MessageBox(
                            "Sorry, the link, \"%s\" is broken, please contact the webmaster" %
                            pipeline_filename,
                            caption="Unable to access pipeline via internet",
                            style=wx.OK | wx.ICON_INFORMATION)
                    return

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
                    wx.MessageBox(
                            'Now that you have loaded an example pipeline, press the "Analyze images" button to access and process a small image set from the CellProfiler website so you can see how CellProfiler works.',
                            '', wx.ICON_INFORMATION)

                wx.CallAfter(fn)
                # try:
                # wx.CallAfter(wx.GetApp().frame.pipeline.load, urllib2.urlopen(pipeline_filename))
                # wx.CallAfter(wx.MessageBox,
                # 'Now that you have loaded an example pipeline, press the "Analyze images" button to access and process a small image set from the CellProfiler website so you can see how CellProfiler works.', '', wx.ICON_INFORMATION)
            except:
                wx.MessageBox(
                        'CellProfiler was unable to load %s' %
                        pipeline_filename, "Error loading pipeline",
                        style=wx.OK | wx.ICON_ERROR)
        else:
            newpage = content.find_link(href)
            if newpage is not None:
                self.SetPage(newpage)
            else:
                super(HtmlClickableWindow, self).OnLinkClicked(linkinfo)

    def OnOpeningURL(self, file_format, url):
        if file_format == wx.html.HTML_URL_IMAGE:
            if url.startswith(MEMORY_SCHEME):
                path = cellprofiler.icons.get_builtin_images_path()
                full_path = os.path.join(path, url[len(MEMORY_SCHEME):])
                if sys.platform.startswith("win"):
                    my_url = full_path
                else:
                    my_url = "file:" + urllib.pathname2url(full_path)
                return my_url
        return wx.html.HTML_OPEN
