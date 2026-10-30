import gc
from unittest.mock import Mock

import pytest
import wx
import wx.grid
import wx.html2
import wx.siplib
from cellprofiler.gui._wx_compat import GridWithLabelRenderersMixin, create_webview


@pytest.fixture(scope="module")
def app():
    if not wx.App.IsDisplayAvailable():
        pytest.skip("A display is required for native wx tests")
    application = wx.GetApp() or wx.App()
    yield application


@pytest.fixture
def frame(app):
    window = wx.Frame(None)
    yield window
    window.Destroy()
    app.ProcessPendingEvents()


class LabelGrid(wx.grid.Grid, GridWithLabelRenderersMixin):
    def __init__(self, parent):
        wx.grid.Grid.__init__(self, parent)
        GridWithLabelRenderersMixin.__init__(self)
        self.CreateGrid(2, 2)


@pytest.mark.parametrize(
    "handler, window_getter, renderer_setter, exposed",
    [
        (
            "_onPaintRowLabels",
            "GetGridRowLabelWindow",
            "SetDefaultRowLabelRenderer",
            "CalcRowLabelsExposed",
        ),
        (
            "_onPaintColLabels",
            "GetGridColLabelWindow",
            "SetDefaultColLabelRenderer",
            "CalcColLabelsExposed",
        ),
        (
            "_onPaintCornerLabel",
            "GetGridCornerLabelWindow",
            "SetCornerLabelRenderer",
            None,
        ),
    ],
)
def test_label_paint_ignores_stale_event_object(
    frame, monkeypatch, handler, window_getter, renderer_setter, exposed
):
    grid = LabelGrid(frame)
    renderer = Mock()
    getattr(grid, renderer_setter)(renderer)
    if exposed:
        monkeypatch.setattr(grid, exposed, lambda *args: [0])
    paint_dc = Mock(side_effect=lambda window: wx.MemoryDC(wx.Bitmap(100, 100)))
    monkeypatch.setattr(wx, "PaintDC", paint_dc)
    event = Mock()
    event.GetEventObject.side_effect = AssertionError("Stale SIP event object")

    getattr(grid, handler)(event)

    event.GetEventObject.assert_not_called()
    paint_dc.assert_called_once_with(getattr(grid, window_getter)())
    renderer.Draw.assert_called_once()


def test_webview_is_owned_by_parent(frame):
    if not any(
        wx.html2.WebView.IsBackendAvailable(backend)
        for backend in (
            wx.html2.WebViewBackendDefault,
            wx.html2.WebViewBackendWebKit,
            wx.html2.WebViewBackendEdge,
        )
    ):
        pytest.skip("No native WebView backend is installed")
    content = create_webview(frame)
    assert content.GetParent() == frame
    assert not wx.siplib.ispyowned(content)
    del content
    gc.collect()
    assert any(isinstance(child, wx.html2.WebView) for child in frame.GetChildren())


@pytest.mark.parametrize("backend", [b"backend", "backend"])
def test_webview_backend_identifiers(frame, monkeypatch, backend):
    content = Mock()
    content.Create.return_value = True
    factory = Mock()
    factory.IsBackendAvailable.return_value = True
    factory.New.return_value = content
    monkeypatch.setattr(wx.html2, "WebView", factory)
    monkeypatch.setattr(wx.html2, "WebViewBackendEdge", backend)

    assert create_webview(frame) is content

    factory.New.assert_called_once_with(backend=backend)
    content.Create.assert_called_once_with(frame)


def test_webview_falls_back_after_failed_creation(frame, monkeypatch):
    failed = Mock()
    failed.Create.return_value = False
    success = Mock()
    success.Create.return_value = True
    factory = Mock()
    factory.IsBackendAvailable.return_value = True
    factory.New.side_effect = [failed, success]
    monkeypatch.setattr(wx.html2, "WebView", factory)

    assert create_webview(frame) is success

    failed.Destroy.assert_called_once()
    assert factory.New.call_count == 2


def test_no_webview_backend(frame, monkeypatch):
    factory = Mock()
    factory.IsBackendAvailable.return_value = False
    monkeypatch.setattr(wx.html2, "WebView", factory)

    with pytest.raises(RuntimeError, match="No usable"):
        create_webview(frame)

    factory.New.assert_not_called()


def test_cursor_and_device_context_apis(frame):
    frame.SetCursor(wx.Cursor(wx.CURSOR_IBEAM))
    frame.SetCursor(wx.NullCursor)
    dc = wx.MemoryDC(wx.Bitmap(100, 100))
    assert len(dc.GetFullTextExtent("metadata")) == 4
    assert len(dc.GetPartialTextExtents("metadata")) == 8
    dc.DrawLabel("metadata", wx.Rect(0, 0, 100, 100))
    dc.SelectObject(wx.NullBitmap)
