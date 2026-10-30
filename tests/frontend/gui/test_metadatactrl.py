from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import wx
from cellprofiler.gui.metadatactrl import MetadataControl


@pytest.fixture
def control(monkeypatch):
    if not wx.App.IsDisplayAvailable():
        pytest.skip("A display is required for native wx tests")
    app = wx.GetApp() or wx.App()
    frame = wx.Frame(None)
    pipeline = SimpleNamespace(get_measurement_columns=lambda module: [])
    ctrl = MetadataControl(pipeline, None, frame, value="abcdef")
    clipboard = Mock()
    clipboard.Open.return_value = True
    clipboard.SetData.return_value = True
    monkeypatch.setattr(wx, "TheClipboard", clipboard)
    yield ctrl, clipboard
    frame.Destroy()
    app.ProcessPendingEvents()


def test_copy_reversed_selection(control):
    ctrl, clipboard = control
    ctrl.selection = [4, 1]
    assert ctrl.CanCopy()
    assert ctrl.CanCut()

    ctrl.Copy()

    assert clipboard.SetData.call_args.args[0].GetText() == "bcd"
    assert ctrl.GetValue() == "abcdef"
    clipboard.Close.assert_called_once()


def test_cut_keeps_text_if_clipboard_fails(control):
    ctrl, clipboard = control
    ctrl.selection = [1, 4]
    clipboard.SetData.return_value = False

    ctrl.Cut()

    assert ctrl.GetValue() == "abcdef"
    clipboard.Close.assert_called_once()


def test_cut_selection(control):
    ctrl, clipboard = control
    ctrl.selection = [1, 4]

    ctrl.Cut()

    assert ctrl.GetValue() == "aef"
    assert not ctrl.CanCopy()
    clipboard.Close.assert_called_once()


def test_cut_reversed_selection_at_cursor(control):
    ctrl, _clipboard = control
    ctrl.move_cursor_pos(1)
    ctrl.selection = [4, 1]

    ctrl.Cut()

    assert ctrl.GetValue() == "aef"
    assert ctrl.selection == [1, 1]
    assert not ctrl.CanCopy()


def test_paste_replaces_selection(control):
    ctrl, clipboard = control
    ctrl.selection = [1, 4]

    def read(data):
        data.SetText("λ🧬")
        return True

    clipboard.GetData.side_effect = read

    ctrl.Paste()

    assert ctrl.GetValue() == "aλ🧬ef"
    assert not ctrl.CanCopy()
    clipboard.Close.assert_called_once()


def test_select_all(control):
    ctrl, clipboard = control

    ctrl.SelectAll()
    ctrl.Copy()

    assert clipboard.SetData.call_args.args[0].GetText() == "abcdef"


def test_clipboard_unavailable(control):
    ctrl, clipboard = control
    clipboard.Open.return_value = False
    ctrl.selection = [0, 6]

    assert not ctrl.CanPaste()
    ctrl.Cut()
    ctrl.Paste()

    assert ctrl.GetValue() == "abcdef"
    clipboard.Close.assert_not_called()


def test_frame_accelerators_dispatch_to_metadata_control(control, monkeypatch):
    from cellprofiler.gui.cpframe import CPFrame

    ctrl, clipboard = control
    monkeypatch.setattr(wx.Window, "FindFocus", lambda: ctrl)

    CPFrame.on_select_all(None)
    CPFrame.on_copy(None)

    assert clipboard.SetData.call_args.args[0].GetText() == "abcdef"
