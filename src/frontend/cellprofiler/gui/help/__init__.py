# coding=utf-8


def make_help_menu(h, window, menu=None):
    import wx
    import cellprofiler.gui.htmldialog
    import cellprofiler.gui.html.utils

    if menu is None:
        menu = wx.Menu()
    if not hasattr(window, "_help_menu_ids"):
        window._help_menu_ids = []
    for key, value in h:
        my_id = wx.NewIdRef()
        window._help_menu_ids.append(my_id)
        if hasattr(value, "__iter__") and not isinstance(value, str):
            menu.Append(my_id, key, make_help_menu(value, window))
        else:

            def show_dialog(event, key=key, value=value):
                value = cellprofiler.gui.html.utils.rst_to_html_fragment(value)
                dlg = cellprofiler.gui.htmldialog.HTMLDialog(window, key, value)
                dlg.Show()

            menu.Append(my_id, key)
            window.Bind(wx.EVT_MENU, show_dialog, id=my_id)

    return menu
