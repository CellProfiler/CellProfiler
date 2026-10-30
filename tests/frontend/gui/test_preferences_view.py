import pytest
import wx
from cellprofiler.gui.constants.preferences_view import WELCOME_MESSAGE
from cellprofiler.gui.preferences_view import PreferencesView
from cellprofiler_core.pipeline import Pipeline
from cellprofiler_core.preferences import report_progress


@pytest.fixture
def view(monkeypatch):
    if not wx.App.IsDisplayAvailable():
        pytest.skip("A display is required for native wx tests")
    app = wx.GetApp() or wx.App()
    frame = wx.Frame(None)
    sizer = wx.BoxSizer(wx.VERTICAL)
    frame.SetSizer(sizer)
    panels = [wx.Panel(frame) for _ in range(3)]
    for panel in panels:
        sizer.Add(panel)
    preferences_view = PreferencesView(sizer, *panels)

    def reject_global_cursor(cursor):
        pytest.fail("Progress must not use the incompatible global wx.SetCursor API")

    monkeypatch.setattr(wx, "SetCursor", reject_global_cursor)
    assert not wx.IsBusy()
    try:
        yield preferences_view
    finally:
        preferences_view.close()
        frame.Destroy()
        app.ProcessPendingEvents()
    assert not wx.IsBusy()


def test_progress_updates_and_completion_restore_cursor(view):
    view.progress_callback("operation", 0, "Adding images")
    assert wx.IsBusy()
    view.progress_callback("operation", 0.5, "Adding images")
    assert wx.IsBusy()

    view.progress_callback("operation", 1, "Done")

    assert not wx.IsBusy()


def test_nested_progress_keeps_cursor_until_outer_completion(view):
    view.progress_callback("outer", 0, "Loading pipeline")
    view.progress_callback("inner", 0, "Adding images")
    view.progress_callback("inner", 1, "Done")
    assert wx.IsBusy()

    view.progress_callback("outer", 1, "Done")

    assert not wx.IsBusy()


def test_progress_reset_after_exception_restores_cursor(view):
    view.progress_callback("operation", 0, "Adding images")
    assert wx.IsBusy()

    view.progress_callback(None, 0, "Reset after exception")
    view.progress_callback(None, 0, "Repeated reset")

    assert not wx.IsBusy()


def test_progress_preserves_an_existing_busy_cursor(view):
    with wx.BusyCursor():
        view.progress_callback("operation", 0, "Adding images")
        view.progress_callback("operation", 1, "Done")
        assert wx.IsBusy()
    assert not wx.IsBusy()


def test_message_only_progress_does_not_change_cursor(view):
    view.progress_callback("operation", None, "Preparing images")
    assert not wx.IsBusy()


def test_closing_view_releases_cursor_and_unregisters_progress(view):
    view.progress_callback("operation", 0, "Adding images")
    assert wx.IsBusy()

    view.close()
    report_progress("later-operation", 0, "Another view is loading images")

    assert not wx.IsBusy()


def test_destroying_view_releases_cursor(view):
    view.progress_callback("operation", 0, "Adding images")
    assert wx.IsBusy()

    view._PreferencesView__odds_and_ends_panel.Destroy()
    wx.GetApp().ProcessPendingEvents()

    assert not wx.IsBusy()


def test_adding_pipeline_images_reports_progress_without_cursor_typeerror(
    view, tmp_path
):
    image = tmp_path / "example.tif"
    image.touch()
    pipeline = Pipeline()

    pipeline.add_pathnames_to_file_list([str(image)])

    assert len(pipeline.file_list) == 1
    assert not wx.IsBusy()
    status = view._PreferencesView__status_text
    assert status.GetLabel() == WELCOME_MESSAGE
