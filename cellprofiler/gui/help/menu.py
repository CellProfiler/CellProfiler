# coding:utf-8

import cellprofiler.gui.help.content
import cellprofiler.gui.help.search
import cellprofiler.gui.htmldialog
import cellprofiler.gui.menu
import cellprofiler.modules


class Menu(cellprofiler.gui.menu.Menu):
    def __init__(self, frame):
        super(Menu, self).__init__(frame)

        self.search_frame = cellprofiler.gui.help.search.Search(self.frame)

        self.append(
            "Show Welcome Screen",
            event_fn=lambda _: self.frame.show_welcome_screen(True)
        )

        # TODO: Requires updated online help manual
        # self.append(
        #     "Online Manual",
        #     event_fn=self.__on_help_online_manual
        # )

        self.AppendSeparator()

        self.append(
            "Why Use CellProfiler?",
            contents=cellprofiler.gui.help.content.read_content("why_use_cellprofiler.rst")
        )

        self.AppendSubMenu(
            self.__navigation_menu(),
            "Navigating The Menu Bar"
        )

        self.AppendSubMenu(
            self.__figure_menu(),
            "Using Module Display Windows"
        )

        self.AppendSubMenu(
            self.__project_menu(),
            cellprofiler.gui.help.content.CREATING_A_PROJECT_CAPTION
        )

        self.append(
            "How To Build A Pipeline",
            contents=cellprofiler.gui.help.content.read_content("pipelines_building.rst")
        )

        self.append(
            "Testing Your Pipeline",
            contents=cellprofiler.gui.help.content.read_content("navigation_test_menu.rst")
        )

        self.append(
            "Running Your Pipeline",
            contents=cellprofiler.gui.help.content.read_content("pipelines_running.rst")
        )

        self.AppendSubMenu(
            self.__output_menu(),
            "Using Your Output"
        )

        self.append(
            "Troubleshooting Memory and Speed Issues",
            contents=cellprofiler.gui.help.content.read_content("other_troubleshooting.rst")
        )

        self.append(
            "Batch Processing",
            contents=cellprofiler.gui.help.content.read_content("other_batch.rst")
        )

        self.AppendSubMenu(
            self.__legacy_menu(),
            "Legacy Modules and Features"
        )

        self.AppendSubMenu(
            self.__other_menu(),
            "Other Features"
        )

        self.AppendSeparator()

        self.append(
            "Developer's Guide",
            event_fn=lambda _: self.__on_help_developers_guide()
        )

        self.append(
            "Source Code",
            event_fn=lambda _: self.__on_help_source_code()
        )

        self.AppendSeparator()

        self.append(
            "Search help...",
            event_fn=lambda _: self.__on_search_help()
        )

    def __figure_menu(self):
        figure_menu = cellprofiler.gui.menu.Menu(self.frame)

        figure_menu.append(
            "Using The Display Window Menu Bar",
            contents=cellprofiler.gui.help.content.read_content("display_menu_bar.rst")
        )

        figure_menu.append(
            "Using The Interactive Navigation Toolbar",
            contents=cellprofiler.gui.help.content.read_content("display_interactive_navigation.rst")
        )

        figure_menu.append(
            "How To Use The Image Tools",
            contents=cellprofiler.gui.help.content.read_content("display_image_tools.rst")
        )

        return figure_menu

    def __legacy_menu(self):
        legacy_menu = cellprofiler.gui.menu.Menu(self.frame)

        legacy_menu.append(
            "Load Modules",
            contents=cellprofiler.gui.help.content.read_content("legacy_load_modules.rst")
        )

        legacy_menu.append(
            "Setting the Output Filename",
            contents=cellprofiler.gui.help.content.read_content("legacy_output_file.rst")
        )

        legacy_menu.append(
            "MATLAB format images",
            contents=cellprofiler.gui.help.content.read_content("legacy_matlab_image.rst")
        )

        return legacy_menu

    def __navigation_menu(self):
        navigation_menu = cellprofiler.gui.menu.Menu(self.frame)

        navigation_menu.append(
            "Using the File Menu",
            contents=cellprofiler.gui.help.content.read_content("navigation_file_menu.rst")
        )

        navigation_menu.append(
            "Using the Edit Menu",
            contents=cellprofiler.gui.help.content.read_content("navigation_edit_menu.rst")
        )

        navigation_menu.append(
            "Using the Test Menu",
            contents=cellprofiler.gui.help.content.read_content("navigation_test_menu.rst")
        )

        navigation_menu.append(
            "Using the Window Menu",
            contents=cellprofiler.gui.help.content.read_content("navigation_window_menu.rst")
        )

        navigation_menu.append(
            "Using the Parameter Sampling Menu",
            contents=cellprofiler.gui.help.content.read_content("navigation_parameter_sampling_menu.rst")
        )

        navigation_menu.append(
            "Using the Data Tools Menu",
            contents=cellprofiler.gui.help.content.read_content("navigation_data_tools_menu.rst")
        )

        return navigation_menu

    @staticmethod
    def __on_help_developers_guide():
        import webbrowser
        webbrowser.open("https://github.com/CellProfiler/CellProfiler/wiki")

    # @staticmethod
    # def __on_help_online_manual(event):
    #     import webbrowser
    #     webbrowser.open("http://d1zymp9ayga15t.cloudfront.net/CPmanual/index.html")

    @staticmethod
    def __on_help_source_code():
        import webbrowser
        webbrowser.open("https://github.com/CellProfiler/CellProfiler")

    def __on_search_help(self):
        if self.search_frame is not None:
            self.search_frame.Show()

            self.search_frame.Raise()

    def __other_menu(self):
        other_menu = cellprofiler.gui.menu.Menu(self.frame)

        other_menu.append(
            "Running Multiple Pipelines",
            contents=cellprofiler.gui.help.content.read_content("other_multiple_pipelines.rst")
        )

        other_menu.append(
            "Configuring Logging",
            contents=cellprofiler.gui.help.content.read_content("other_logging.rst")
        )

        other_menu.append(
            "Accessing Images From OMERO",
            contents=cellprofiler.gui.help.content.read_content("other_omero.rst")
        )
        
        other_menu.append(
            "Using Plugins",
            contents=cellprofiler.gui.help.content.read_content("other_plugins.rst")
        )


        return other_menu

    def __output_menu(self):
        output_menu = cellprofiler.gui.menu.Menu(self.frame)

        output_menu.append(
            "How Measurements are Named",
            contents=cellprofiler.gui.help.content.read_content("output_measurements.rst")
        )

        output_menu.append(
            "Using Spreadsheets and Databases",
            contents=cellprofiler.gui.help.content.read_content("output_spreadsheets.rst")
        )

        output_menu.append(
            "Plate Viewer",
            contents=cellprofiler.gui.help.content.read_content("output_plateviewer.rst")
        )

        return output_menu

    def __project_menu(self):
        project_menu = cellprofiler.gui.menu.Menu(self.frame)

        project_menu.append(
            "Introduction to Projects",
            contents=cellprofiler.gui.help.content.read_content("projects_introduction.rst")
        )

        project_menu.append(
            "Selecting Images for Input",
            contents=cellprofiler.gui.help.content.read_content("projects_selecting_images.rst")
        )

        project_menu.append(
            "Configuring Images for Analysis",
            contents=cellprofiler.gui.help.content.read_content("projects_configure_images.rst")
        )

        project_menu.append(
            "Loading Image Stacks and Movies",
            contents=cellprofiler.gui.help.content.read_content("projects_image_sequences.rst")
        )

        return project_menu
