import wx
from rapidfuzz.utils import default_process
from rapidfuzz.process import extract
from rapidfuzz.distance import DamerauLevenshtein

from cellprofiler_core.utilities.core.modules import (
    get_module_names,
    instantiate_module,
    get_module_class,
)

import cellprofiler.gui
import cellprofiler.gui.cpframe
import cellprofiler.gui.help.search
import cellprofiler.gui.utilities.icon
import cellprofiler.modules


class AddModuleFrame(wx.Frame):
    """The window frame that lets you add modules to a pipeline

    """

    def __init__(self, *args, **kwds):
        """Initialize the frame and its layout

        """
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        # Top level panels
        left_panel = wx.Panel(self, -1)
        right_panel = wx.Panel(self, -1)
        # Module categories (in left panel)
        module_categories_text = wx.StaticText(
            left_panel, -1, "Module Categories", style=wx.ALIGN_CENTER
        )
        font = module_categories_text.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        module_categories_text.SetFont(font)
        self.__module_categories_list_box = wx.ListBox(left_panel, -1)
        # Control panel for the selected module
        selected_module_panel = wx.Panel(left_panel, -1)
        add_to_pipeline_button = wx.Button(
            selected_module_panel, -1, "+ Add to Pipeline"
        )
        module_help_button = wx.Button(selected_module_panel, -1, "? Module Help")
        # Other buttons
        getting_started_button = wx.Button(left_panel, -1, "Getting Started")
        done_button = wx.Button(left_panel, -1, "Done")
        # Right-side panel
        self.__module_list_box = wx.ListBox(right_panel, -1)
        w, h, _, _ = self.__module_list_box.GetFullTextExtent(
            "MeasureObjectIntensityDistribution"
        )
        self.__module_list_box.SetMinSize(wx.Size(w, h * 30))
        # Sizers
        self.SetSizer(wx.BoxSizer(wx.VERTICAL))
        search_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.GetSizer().Add(search_sizer, 0, wx.EXPAND | wx.ALL, 2)
        search_sizer.Add(
            wx.StaticText(self, label="Find Modules:"),
            0,
            wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL,
        )
        self.search_text = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        search_sizer.Add(self.search_text, 1, wx.EXPAND)
        self.search_button = wx.Button(self, label="Search Help")
        search_sizer.Add(self.search_button, 0, wx.EXPAND)
        self.GetSizer().AddSpacer(2)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.AddMany(
            [(left_panel, 0, wx.EXPAND | wx.LEFT, 5), (right_panel, 1, wx.EXPAND)]
        )
        self.GetSizer().Add(top_sizer, 1, wx.EXPAND)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_sizer.Add(module_categories_text, 0, wx.EXPAND)
        left_sizer.AddSpacer(4)
        left_sizer.Add(self.__module_categories_list_box, 1, wx.EXPAND)
        left_sizer.Add((-1, 10))
        left_sizer.Add(selected_module_panel, 0, wx.EXPAND)
        left_sizer.Add((-1, 10))
        left_sizer.Add(getting_started_button, 0, wx.EXPAND)
        left_sizer.AddSpacer(2)
        left_sizer.Add(done_button, 0, wx.EXPAND | wx.BOTTOM, 5)
        left_panel.SetSizer(left_sizer)

        right_sizer = wx.BoxSizer(wx.VERTICAL)
        right_sizer.Add(self.__module_list_box, 1, wx.EXPAND | wx.ALL, 5)
        right_panel.SetSizer(right_sizer)

        selected_module_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        selected_module_panel_sizer.Add(add_to_pipeline_button, 0, wx.EXPAND)
        selected_module_panel_sizer.AddSpacer(2)
        selected_module_panel_sizer.Add(module_help_button, 0, wx.EXPAND)
        selected_module_panel.SetSizer(selected_module_panel_sizer)

        self.__set_icon()
        accelerators = wx.AcceleratorTable(
            [(wx.ACCEL_CMD, ord("W"), cellprofiler.gui.cpframe.ID_FILE_EXIT)]
        )
        self.SetAcceleratorTable(accelerators)

        self.Bind(wx.EVT_CLOSE, self.__on_close, self)
        self.Bind(
            wx.EVT_LISTBOX,
            self.__on_category_selected,
            self.__module_categories_list_box,
        )
        self.Bind(
            wx.EVT_LISTBOX_DCLICK, self.__on_add_to_pipeline, self.__module_list_box
        )
        self.Bind(wx.EVT_BUTTON, self.__on_add_to_pipeline, add_to_pipeline_button)
        self.Bind(wx.EVT_BUTTON, self.__on_close, done_button)
        self.Bind(wx.EVT_BUTTON, self.__on_help, module_help_button)
        self.Bind(wx.EVT_BUTTON, self.__on_getting_started, getting_started_button)
        self.Bind(
            wx.EVT_MENU, self.__on_close, id=cellprofiler.gui.cpframe.ID_FILE_EXIT
        )
        self.Bind(wx.EVT_CHAR_HOOK, self.__on_special_key)
        self.search_text.Bind(wx.EVT_TEXT, self.__on_search_modules)
        self.search_button.Bind(wx.EVT_BUTTON, self.__on_search_help)
        self.__get_module_files()
        self.__set_categories()
        self.__listeners = []
        self.__module_categories_list_box.Select(0)
        self.__on_category_selected(None)
        self.Fit()
        self.search_text.SetFocus()

    def __on_close(self, event):
        self.Hide()
        self.search_text.SetFocus()
        self.search_text.SelectAll()

    def __set_icon(self):
        icon = cellprofiler.gui.utilities.icon.get_cp_icon()
        self.SetIcon(icon)

    def __get_module_files(self):
        self.__module_files = [
            "File Processing",
            "Image Processing",
            "Object Processing",
            "Measurement",
            "Data Tools",
            "Other",
            "All",
        ]
        self.__module_dict = {}
        for key in self.__module_files:
            self.__module_dict[key] = {}

        for mn in get_module_names():

            def loader(module_num, mn=mn):
                module = instantiate_module(mn)
                module.set_module_num(module_num)
                return module

            try:
                module = get_module_class(mn)
                if module.is_input_module():
                    continue
                categories = (
                    [module.category]
                    if isinstance(module.category, str)
                    else list(module.category)
                ) + ["All"]
                for category in categories:
                    if category not in self.__module_files:
                        self.__module_files.insert(-2, category)
                        self.__module_dict[category] = {}
                    self.__module_dict[category][module.module_name] = loader
            except Exception as e:
                import traceback
                import logging

                logging.root.error(
                    "Unable to instantiate module %s.\n\n", mn, exc_info=True
                )

    def __set_categories(self):
        self.__module_categories_list_box.AppendItems(self.__module_files)

    def __on_category_selected(self, event):
        self.__module_list_box.Enable(True)
        category = self.__get_selected_category()
        self.__module_list_box.Clear()
        keys = list(self.__module_dict[category].keys())
        sorted(keys)
        self.__module_list_box.AppendItems(keys)
        self.__module_list_box.Select(0)

    def __get_selected_category(self):
        return self.__module_files[self.__module_categories_list_box.GetSelection()]

    def __on_add_to_pipeline(self, event):
        category = self.__get_selected_category()
        idx = self.__module_list_box.GetSelection()
        if idx != wx.NOT_FOUND:
            filename = self.__module_list_box.GetItems()[idx]
            self.notify(
                AddToPipelineEvent(filename, self.__module_dict[category][filename])
            )

    def __on_help(self, event):
        category = self.__get_selected_category()
        idx = self.__module_list_box.GetSelection()
        if idx != wx.NOT_FOUND:
            filename = self.__module_list_box.GetItems()[idx]
            loader = self.__module_dict[category][filename]
            module = loader(0)
            if isinstance(self.GetParent(), cellprofiler.gui.cpframe.CPFrame):
                self.GetParent().do_help_module(module.module_name, module.get_help())
            else:
                help_text = module.get_help()
                wx.MessageBox(help_text)

    def __on_search_help(self, event):
        if len(self.search_text.GetValue()) == 0:
            wx.MessageBox(
                "Please enter the search text to be found.",
                caption="No text to search",
                parent=self,
                style=wx.OK | wx.CENTRE | wx.ICON_INFORMATION,
            )
            self.search_text.SetFocus()
            return

        html = cellprofiler.gui.help.search.search_module_help(
            self.search_text.GetValue()
        )
        if html is None:
            wx.MessageBox(
                'No references found for "%s".' % self.search_text.GetValue(),
                caption="Text not found",
                parent=self,
                style=wx.OK | wx.CENTRE | wx.ICON_INFORMATION,
            )
        else:
            self.display_helpframe(
                html, 'Help matching "%s"' % self.search_text.GetValue()
            )

    def __on_search_modules(self, event):
        self.__module_list_box.Enable(True)
        if (
            len(self.search_text.GetValue()) == 0
            or self.__module_categories_list_box.GetSelection() != -1
        ):
            self.__module_categories_list_box.Select(-1)
            self.__on_category_selected(None)

        keys = list(self.__module_dict["All"].keys())

        # results in [('Module1', score1, idx1), ('Module2', score2, idx2), ...]
        # where idx is the index of the result in `keys`
        top_scorers = [m[0] for m in extract(
            self.search_text.GetValue(),
            keys,
            processor=default_process,
            scorer=DamerauLevenshtein.similarity,
            limit=10
        )]

        self.__module_list_box.Clear()
        self.__module_list_box.AppendItems(top_scorers)
        if len(top_scorers) > 0:
            self.__module_list_box.Select(0)
        else:
            self.__module_list_box.AppendItems("No matching modules")
            self.__module_list_box.Enable(False)

    def __on_special_key(self, event):
        # Capture keyboard shortcuts
        key = event.GetKeyCode()
        numitems = len(self.__module_list_box.GetItems())
        if key == wx.WXK_ESCAPE:
            self.Close()
            return
        elif key in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER):
            self.__on_add_to_pipeline(event)
            return
        elif numitems <= 1:
            # No point moving selector
            pass
        elif key == wx.WXK_DOWN:
            i = self.__module_list_box.GetSelection()
            self.__module_list_box.Select(min(i + 1, numitems - 1))
            return
        elif key == wx.WXK_UP:
            i = self.__module_list_box.GetSelection()
            self.__module_list_box.Select(max(0, i - 1))
            return
        event.Skip()

    def __on_getting_started(self, event):
        import cellprofiler.gui.help.content
        import cellprofiler.gui.html.utils

        self.display_helpframe(
            cellprofiler.gui.html.utils.rst_to_html_fragment(
                cellprofiler.gui.help.content.read_content("pipelines_building.rst")
            ),
            "Add modules: Getting Started",
        )

    def display_helpframe(self, help_text, title):
        from cellprofiler.gui.html.htmlwindow import HtmlClickableWindow

        helpframe = wx.Frame(self, -1, title, size=(640, 480))
        sizer = wx.BoxSizer()
        helpframe.SetSizer(sizer)
        window = HtmlClickableWindow(helpframe)
        sizer.Add(window, 1, wx.EXPAND)
        window.AppendToPage(help_text)
        helpframe.SetIcon(cellprofiler.gui.utilities.icon.get_cp_icon())
        helpframe.Layout()
        helpframe.Show()

    def add_listener(self, listener):
        self.__listeners.append(listener)

    def remove_listener(self, listener):
        self.__listeners.remove(listener)

    def notify(self, event):
        for listener in self.__listeners:
            listener(self, event)


class AddToPipelineEvent(object):
    def __init__(self, module_name, module_loader):
        self.module_name = module_name
        self.__module_loader = module_loader

    def get_module_loader(self):
        """Return a function that, when called, will produce a module

        The function takes one argument: the module number
        """
        return self.__module_loader

    module_loader = property(get_module_loader)
