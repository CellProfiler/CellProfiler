import wx
import wx.html
import content
import webbrowser
import urllib2
import cellprofiler.preferences as cpprefs
from cellprofiler.icons import get_builtin_images_path

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
                self.SetPage('')
                self.BackgroundColour = cpprefs.get_background_color()
        elif href.startswith('load:'):
            try:
                wx.CallAfter(wx.GetApp().frame.pipeline.load, urllib2.urlopen(href[5:]))
            except:
                wx.MessageBox(
                    'CellProfiler was unable to load href[5:]' %
                    options.pipeline_filename, "Error loading pipeline",
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
            if url.startswith('memory:'):
                return url.replace('memory:', 'file://' + get_builtin_images_path())
        return wx.html.HTML_OPEN

if __name__ == '__main__':
    app = wx.App(0)
    frame = wx.Frame(None, -1, 'foo', (500, 500))
    htmlwin = HtmlClickableWindow(frame, wx.ID_ANY, style=wx.NO_BORDER)
    frame.Show(True)
    frame.Center()
    app.MainLoop()
