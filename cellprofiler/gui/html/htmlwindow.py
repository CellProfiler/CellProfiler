# coding=utf-8

import os
import sys
import webbrowser

import six.moves.urllib.request
import wx.html

import cellprofiler.icons
from cellprofiler.gui.html import utils

MEMORY_SCHEME = "memory:"


class HtmlClickableWindow(wx.html.HtmlWindow):
    def __init__(self, *args, **kwargs):
        wx.html.HtmlWindow.__init__(self, *args, **kwargs)

    def OnLinkClicked(self, linkinfo):
        href = linkinfo.Href
        if href.startswith("#"):
            super(HtmlClickableWindow, self).OnLinkClicked(linkinfo)
        elif href.startswith('http://') or href.startswith("https://"):
            webbrowser.open(href)
        else:
            newpage = utils.find_link(href)
            if newpage is not None:
                self.SetPage(newpage)
            else:
                super(HtmlClickableWindow, self).OnLinkClicked(linkinfo)

    def OnOpeningURL(self, file_format, url, **kwargs):
        if file_format == wx.html.HTML_URL_IMAGE:
            if url.startswith(MEMORY_SCHEME):
                path = cellprofiler.icons.get_builtin_images_path()
                full_path = os.path.join(path, url[len(MEMORY_SCHEME):])
                if sys.platform.startswith("win"):
                    my_url = full_path
                else:
                    my_url = "file:" + six.moves.urllib.request.pathname2url(full_path)
                return my_url
        return wx.html.HTML_OPEN
