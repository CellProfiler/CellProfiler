import wx
import wx.html
import content
import webbrowser
import cellprofiler.preferences as cpprefs

class HtmlClickableWindow(wx.html.HtmlWindow):
    def __init__(self, *args, **kwargs):
        wx.html.HtmlWindow.__init__(self, *args, **kwargs)
        self.HTMLBackgroundColour = cpprefs.get_background_color()
        self.OnLinkClicked(wx.html.HtmlLinkInfo('startup_main', ''))

    def OnLinkClicked(self, linkinfo):
        href = linkinfo.Href
        if href.startswith('http://'):
            webbrowser.open(href)
        elif href.startswith('pref:'):
            if 'no_display' in href:
                cpprefs.set_startup_blurb(False)
                self.SetPage('')
                self.BackgroundColour = cpprefs.get_background_color()
        else:
            newpage = content.find_link(href)
            if newpage is not None:
                self.SetPage(newpage)
                self.BackgroundColour = cpprefs.get_background_color()
            else:
                wx.html.HtmlWindow.OnLinkClicked(self, linkinfo)

if __name__ == '__main__':
    app = wx.App(0)
    frame = wx.Frame(None, -1, 'foo', (500, 500))
    htmlwin = HtmlClickableWindow(frame, wx.ID_ANY, style=wx.NO_BORDER)
    frame.Show(True)
    frame.Center()
    app.MainLoop()
