import os
import re

import wx
import wx.html

import cellprofiler.gui
import cellprofiler.gui.help.content
import cellprofiler.gui.html.utils
import cellprofiler.modules

MENU_HELP = {
    "Accessing Images From OMERO": cellprofiler.gui.help.content.read_content("other_omero.rst"),
    "Batch Processing": cellprofiler.gui.help.content.read_content("other_batch.rst"),
    "How To Build A Pipeline": cellprofiler.gui.help.content.read_content("pipelines_building.rst"),
    "Configuring Images for Analysis": cellprofiler.gui.help.content.read_content("projects_configure_images.rst"),
    "Configuring Logging": cellprofiler.gui.help.content.read_content("other_logging.rst"),
    "Introduction to Projects": cellprofiler.gui.help.content.read_content("projects_introduction.rst"),
    "Load Modules": cellprofiler.gui.help.content.read_content("legacy_load_modules.rst"),
    "Loading Image Stacks and Movies": cellprofiler.gui.help.content.read_content("projects_image_sequences.rst"),
    "MATLAB format images": cellprofiler.gui.help.content.read_content("legacy_matlab_image.rst"),
    "How Measurements are Named": cellprofiler.gui.help.content.read_content("output_measurements.rst"),
    "Troubleshooting Memory and Speed Issues": cellprofiler.gui.help.content.read_content("other_troubleshooting.rst"),
    "Using the Data Tools Menu": cellprofiler.gui.help.content.read_content("navigation_data_tools_menu.rst"),
    "Using the Edit Menu": cellprofiler.gui.help.content.read_content("navigation_file_menu.rst"),
    "Using the File Menu": cellprofiler.gui.help.content.read_content("navigation_edit_menu.rst"),
    "Using the Window Menu": cellprofiler.gui.help.content.read_content("navigation_window_menu.rst"),
    "How To Use The Image Tools": cellprofiler.gui.help.content.read_content("display_image_tools.rst"),
    "Using The Interactive Navigation Toolbar":
        cellprofiler.gui.help.content.read_content("display_interactive_navigation.rst"),
    "Using The Display Window Menu Bar": cellprofiler.gui.help.content.read_content("display_menu_bar.rst"),
    "Using the Parameter Sampling Menu": cellprofiler.gui.help.content.read_content(
        "navigation_parameter_sampling_menu.rst"
    ),
    "Plate Viewer": cellprofiler.gui.help.content.read_content("output_plateviewer.rst"),
    "Running Multiple Pipelines": cellprofiler.gui.help.content.read_content("other_multiple_pipelines.rst"),
    "Running Your Pipeline": cellprofiler.gui.help.content.read_content("pipelines_running.rst"),
    "Selecting Images for Input": cellprofiler.gui.help.content.read_content("projects_selecting_images.rst"),
    "Using Spreadsheets and Databases": cellprofiler.gui.help.content.read_content("output_spreadsheets.rst"),
    "Using the Test Menu": cellprofiler.gui.help.content.read_content("navigation_test_menu.rst"),
    "Using Plugins": cellprofiler.gui.help.content.read_content("other_plugins.rst"),
    "Setting the Output Filename": cellprofiler.gui.help.content.read_content("legacy_output_file.rst"),
    "Why Use CellProfiler?": cellprofiler.gui.help.content.read_content("why_use_cellprofiler.rst")
}


class Search(wx.Frame):
    def __init__(self, parent):
        super(Search, self).__init__(
            parent,
            size=(
                wx.SystemSettings.GetMetric(wx.SYS_SCREEN_X) / 2,
                wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y) / 2
            ),
            style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL,
            title="Search CellProfiler help"
        )

        self.AutoLayout = True

        self.Sizer = wx.BoxSizer(wx.VERTICAL)

        search_panel = self.__create_search_panel()
        self.Sizer.Add(search_panel, 0, wx.EXPAND | wx.ALL, 4)

        self.results_view = self.__create_results_view()
        self.Sizer.Add(self.results_view, 1, wx.EXPAND | wx.ALL, 4)

        self.Bind(wx.EVT_CLOSE, self.__on_close)

        self.Layout()

        self.SetIcon(cellprofiler.gui.get_cp_icon())

    def __create_results_view(self):
        html_window = cellprofiler.gui.html.htmlwindow.HtmlClickableWindow(self)

        html_window.Bind(
            wx.html.EVT_HTML_LINK_CLICKED,
            lambda event: self.__on_link_clicked(event, html_window)
        )

        return html_window

    def __create_search_panel(self):
        def __on_search(results_view):
            search_text = search_text_ctrl.Value

            html = search_module_help(search_text)

            if html is None:
                no_results_message = u"""\
<html>
<head>
    <title>"{search_text}" not found in help</title>
</head>
<body>
    <header></header>Could not find "{search_text}" in CellProfiler's help documentation
</body>
</html>
        """.format(**{
                    "search_text": search_text
                })

                results_view.SetPage(no_results_message)
            else:
                results_view.SetPage(html)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        sizer.Add(
            wx.StaticText(self, label="Search:"),
            0,
            wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL
        )

        sizer.AddSpacer(2)

        search_text_ctrl = wx.TextCtrl(self)

        sizer.Add(search_text_ctrl, 1, wx.EXPAND)

        search_button = wx.Button(self, label="Search")

        search_button.SetDefault()

        search_button.Bind(
            wx.EVT_BUTTON,
            lambda _: __on_search(self.results_view)
        )

        sizer.AddSpacer(2)

        sizer.Add(search_button, 0, wx.EXPAND)

        return sizer

    def __on_close(self, event):
        self.Hide()

        event.Veto()

    def __on_link_clicked(self, event, html_window):
        """Handle anchor clicks manually

        The HTML window (on Windows at least) jams the anchor to the
        top of the window which obscures it.
        """
        linkinfo = event.GetLinkInfo()

        if linkinfo.GetHref()[0] != "#":
            event.Skip()

            return

        html_window.ScrollToAnchor(linkinfo.GetHref()[1:])

        html_window.ScrollLines(-1)


def __search_fn(html, text):
    """
    Find the beginning and ending indices of case insensitive matches of "text"
    within the text-data of the HTML, searching only in its body and excluding
    text in the HTML tags.

    :param html: an HTML document
    :param text: a search string
    :return:
    """
    start_match = re.search(r"<\s*body[^>]*?>", html, re.IGNORECASE)

    if start_match is None:
        start = 0
    else:
        start = start_match.end()

    end_match = re.search(r"<\\\s*body", html, re.IGNORECASE)

    if end_match is None:
        end = len(html)
    else:
        end = end_match.start()

    escaped_text = re.escape(text)

    if " " in escaped_text:
        #
        # Many problems here:
        # <b>Groups</b> module
        # Some\ntext
        #
        # For now, just solve the multiple space problems
        #
        escaped_text = escaped_text.replace("\\ ", "\\s+")

    pattern = "(<[^>]*?>|%s)" % escaped_text

    return [(x.start() + start, x.end() + start) for x in re.finditer(
        pattern, html[start:end],
        re.IGNORECASE
    ) if x.group(1)[0] != '<']


def search_module_help(text):
    """
    Search the help for a string

    :param text: find text in the module help using case-insensitive matching
    :return: an html document of all the module help pages that matched or None if no match found.
    """
    matching_help = []

    count = 0

    for menu_item, help_text in MENU_HELP.iteritems():
        help_text = cellprofiler.gui.html.utils.rst_to_html_fragment(help_text)

        matches = __search_fn(help_text, text)

        if len(matches) > 0:
            matching_help.append((menu_item, help_text, matches))

            count += len(matches)

    for module_name in cellprofiler.modules.get_module_names():
        module = cellprofiler.modules.instantiate_module(module_name)

        location = os.path.split(module.create_settings.im_func.func_code.co_filename)[0]

        if location == cellprofiler.preferences.get_plugin_directory():
            continue

        help_text = module.get_help()

        matches = __search_fn(help_text, text)

        if len(matches) > 0:
            matching_help.append((module_name, help_text, matches))

            count += len(matches)

    if len(matching_help) == 0:
        return None

    top = u"""\
<html style="font-family:arial">
<head>
    <title>{count} match{es} found</title>
</head>
<body>
    <h1>Match{es} found</h1><br>
    <ul></ul>
</body>
</html>
""".format(**{
        "count": count,
        "es": u"" if count == 1 else u"es"
    })

    body = u"<br>"

    match_num = 1

    prev_link = u"""\
<a href="#match%d" title="Previous match">
    <img alt="previous match" src=memory:previous.png">
</a>"""

    anchor = u"""<a name="match%d"><u>%s</u></a>"""

    next_link = u"""<a href="#match%d" title="Next match"><img src="memory:next.png" alt="next match"></a>"""

    for title, help_text, pairs in matching_help:
        top += u"""<li><a href="#match{:d}">{}</a></li>\n""".format(match_num, title)

        if help_text.find("<h1>") == -1:
            body += u"<h1>{}</h1>".format(title)

        start_match = re.search(r"<\s*body[^>]*?>", help_text, re.IGNORECASE)

        if start_match is None:
            start = 0
        else:
            start = start_match.end()

        end_match = re.search(r"<\\\s*body", help_text, re.IGNORECASE)

        if end_match is None:
            end = len(help_text)
        else:
            end = end_match.start()

        for begin_pos, end_pos in pairs:
            body += help_text[start:begin_pos]

            if match_num > 1:
                body += prev_link % (match_num - 1)

            body += anchor % (match_num, help_text[begin_pos:end_pos])

            if match_num != count:
                body += next_link % (match_num + 1)

            start = end_pos

            match_num += 1

        body += help_text[start:end] + u"<br>"

    result = u"{}</ul><br>\n{}</body></html>".format(top, body)

    return result
