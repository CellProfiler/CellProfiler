"""Compatibility with wxPython 4.2.5 through 4.3.2."""

import logging
from types import SimpleNamespace

import wx.lib.mixins.gridlabelrenderer as wxglr

LOGGER = logging.getLogger(__name__)


class GridWithLabelRenderersMixin(wxglr.GridWithLabelRenderersMixin):
    """Use known label windows to avoid stale SIP paint-event objects (#2929)."""

    def _onPaintRowLabels(self, event):
        super()._onPaintRowLabels(
            SimpleNamespace(GetEventObject=self.GetGridRowLabelWindow)
        )

    def _onPaintColLabels(self, event):
        super()._onPaintColLabels(
            SimpleNamespace(GetEventObject=self.GetGridColLabelWindow)
        )

    def _onPaintCornerLabel(self, event):
        super()._onPaintCornerLabel(
            SimpleNamespace(GetEventObject=self.GetGridCornerLabelWindow)
        )


def create_webview(parent):
    """Use two-step creation to transfer ownership via Create, avoiding #2970."""
    import wx.html2

    for name in (
        "WebViewBackendEdge",
        "WebViewBackendIE",
        "WebViewBackendWebKit",
        "WebViewBackendDefault",
    ):
        backend = getattr(wx.html2, name)
        if wx.html2.WebView.IsBackendAvailable(backend):
            LOGGER.debug("Using WebView backend: %s (%r)", name, backend)
            content = wx.html2.WebView.New(backend=backend)
            if content is None:
                continue
            if content.Create(parent):
                return content
            content.Destroy()
    raise RuntimeError("No usable wx.html2.WebView backend is available")
