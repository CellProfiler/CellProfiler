# coding:utf-8

import webbrowser
import wx.adv

from cellprofiler.gui.dialog import AboutDialogInfo
import cellprofiler.gui.help.content
import cellprofiler.gui.help.search
import cellprofiler.gui.htmldialog
import cellprofiler.gui.menu


class Menu(cellprofiler.gui.menu.Menu):
    def __init__(self, frame):
        super(Menu, self).__init__(frame)

        self.search_frame = cellprofiler.gui.help.search.Search(self.frame)

        self.append(
            "Show Welcome Screen",
            event_fn=lambda _: self.frame.show_welcome_screen(True),
        )

        self.append("Search help...", event_fn=lambda _: self.__on_search_help())

        self.append("Online Manual", event_fn=self.__on_help_online_manual)

        self.AppendSeparator()

        self.append(
            "Why Use CellProfiler?",
            contents=cellprofiler.gui.help.content.read_content(
                "why_use_cellprofiler.rst"
            ),
        )

        self.AppendSubMenu(self.__navigation_menu(), "Navigating The Menu Bar")

        self.AppendSubMenu(self.__figure_menu(), "Using Module Display Windows")

        self.AppendSubMenu(
            self.__project_menu(),
            cellprofiler.gui.help.content.CREATING_A_PROJECT_CAPTION,
        )

        self.append(
            "How To Build A Pipeline",
            contents=cellprofiler.gui.help.content.read_content(
                "pipelines_building.rst"
            ),
        )

        self.append(
            "Testing Your Pipeline",
            contents=cellprofiler.gui.help.content.read_content(
                "navigation_test_menu.rst"
            ),
        )

        self.append(
            "Running Your Pipeline",
            contents=cellprofiler.gui.help.content.read_content(
                "pipelines_running.rst"
            ),
        )

        self.AppendSubMenu(self.__output_menu(), "Using Your Output")

        self.append(
            "Troubleshooting Memory and Speed Issues",
            contents=cellprofiler.gui.help.content.read_content(
                "other_troubleshooting.rst"
            ),
        )

        self.append(
            "Identifying 3D objects",
            contents=cellprofiler.gui.help.content.read_content(
                "other_3d_identify.rst"
            ),
        )

        self.append(
            "Batch Processing",
            contents=cellprofiler.gui.help.content.read_content("other_batch.rst"),
        )

        self.AppendSubMenu(self.__legacy_menu(), "Legacy Modules and Features")

        self.AppendSubMenu(self.__other_menu(), "Other Features")

        self.AppendSeparator()

        self.append(
            "Developer's Guide", event_fn=lambda _: self.__on_help_developers_guide()
        )

        self.append("Source Code", event_fn=lambda _: self.__on_help_source_code())

        self.AppendSeparator()

        self.append("Check for updates", event_fn=self.find_update)

        self.append("About CellProfiler", event_fn=lambda _: self.about())

    def about(self):
        info = AboutDialogInfo()
        wx.adv.AboutBox(info)
        if wx.GetKeyState(wx.WXK_SHIFT):
            from wx.py.shell import ShellFrame
            cpapp = wx.GetApp()
            if cpapp:
                cpapp = cpapp.frame
                locs = {'app': cpapp, 'pipeline': cpapp.pipeline, 'workspace': cpapp.workspace}
            else:
                locs = None
            s = ShellFrame(self.frame,
                           title="CellProfiler Shell",
                           locals=locs,
                           )
            s.SetStatusText("CellProfiler Debug Interpeter - Use 'app', 'pipeline' and 'workspace' to inspect objects")
            s.Show()

    def find_update(self, event):
        from cellprofiler.gui.checkupdate import check_update
        check_update(self.frame, force=True)

    def __figure_menu(self):
        figure_menu = cellprofiler.gui.menu.Menu(self.frame)

        figure_menu.append(
            "Using The Display Window Menu Bar",
            contents=cellprofiler.gui.help.content.read_content("display_menu_bar.rst"),
        )

        figure_menu.append(
            "Using The Interactive Navigation Toolbar",
            contents=cellprofiler.gui.help.content.read_content(
                "display_interactive_navigation.rst"
            ),
        )

        figure_menu.append(
            "How To Use The Image Tools",
            contents=cellprofiler.gui.help.content.read_content(
                "display_image_tools.rst"
            ),
        )

        return figure_menu

    def __legacy_menu(self):
        legacy_menu = cellprofiler.gui.menu.Menu(self.frame)

        legacy_menu.append(
            "MATLAB format images",
            contents=cellprofiler.gui.help.content.read_content(
                "legacy_matlab_image.rst"
            ),
        )

        return legacy_menu

    def __navigation_menu(self):
        navigation_menu = cellprofiler.gui.menu.Menu(self.frame)

        navigation_menu.append(
            "Using the File Menu",
            contents=cellprofiler.gui.help.content.read_content(
                "navigation_file_menu.rst"
            ),
        )

        navigation_menu.append(
            "Using the Edit Menu",
            contents=cellprofiler.gui.help.content.read_content(
                "navigation_edit_menu.rst"
            ),
        )

        navigation_menu.append(
            "Using the Test Menu",
            contents=cellprofiler.gui.help.content.read_content(
                "navigation_test_menu.rst"
            ),
        )

        navigation_menu.append(
            "Using the Window Menu",
            contents=cellprofiler.gui.help.content.read_content(
                "navigation_window_menu.rst"
            ),
        )

        return navigation_menu

    @staticmethod
    def __on_help_developers_guide():
        webbrowser.open("https://github.com/CellProfiler/CellProfiler/wiki")

    @staticmethod
    def __on_help_online_manual(event):
        webbrowser.open(cellprofiler.gui.help.content.MANUAL_URL)

    @staticmethod
    def __on_help_source_code():
        webbrowser.open("https://github.com/CellProfiler/CellProfiler")

    def __on_search_help(self):
        if self.search_frame is not None:
            self.search_frame.Show()

            self.search_frame.Raise()

    def __other_menu(self):
        other_menu = cellprofiler.gui.menu.Menu(self.frame)

        other_menu.append(
            "Configuring Logging",
            contents=cellprofiler.gui.help.content.read_content("other_logging.rst"),
        )

        #TODO: disabled until CellProfiler/CellProfiler#4684 is resolved
        # other_menu.append(
        #     "Accessing Images From OMERO",
        #     contents=cellprofiler.gui.help.content.read_content("other_omero.rst"),
        # )

        other_menu.append(
            "Using Plugins",
            contents=cellprofiler.gui.help.content.read_content("other_plugins.rst"),
        )

        other_menu.append(
            "Debug Shell",
            contents=cellprofiler.gui.help.content.read_content("other_shell.rst"),
        )

        other_menu.append(
            "Widget Inspector",
            contents=cellprofiler.gui.help.content.read_content("other_widget_inspector.rst"),
        )

        return other_menu

    def __output_menu(self):
        output_menu = cellprofiler.gui.menu.Menu(self.frame)

        output_menu.append(
            "How Measurements are Named",
            contents=cellprofiler.gui.help.content.read_content(
                "output_measurements.rst"
            ),
        )

        output_menu.append(
            "Using Spreadsheets and Databases",
            contents=cellprofiler.gui.help.content.read_content(
                "output_spreadsheets.rst"
            ),
        )

        output_menu.append(
            "Plate Viewer",
            contents=cellprofiler.gui.help.content.read_content(
                "output_plateviewer.rst"
            ),
        )

        return output_menu

    def __project_menu(self):
        project_menu = cellprofiler.gui.menu.Menu(self.frame)

        project_menu.append(
            "Introduction to Projects",
            contents=cellprofiler.gui.help.content.read_content(
                "projects_introduction.rst"
            ),
        )

        project_menu.append(
            "Selecting Images for Input",
            contents=cellprofiler.gui.help.content.read_content(
                "projects_selecting_images.rst"
            ),
        )

        project_menu.append(
            "Configuring Images for Analysis",
            contents=cellprofiler.gui.help.content.read_content(
                "projects_configure_images.rst"
            ),
        )

        project_menu.append(
            "Loading Image Stacks and Movies",
            contents=cellprofiler.gui.help.content.read_content(
                "projects_image_sequences.rst"
            ),
        )

        project_menu.append(
            "Image Ordering",
            contents=cellprofiler.gui.help.content.read_content(
                "projects_image_ordering.rst"
            ),
        )

        return project_menu
