import cellprofiler.gui.figure


class Workspace(cellprofiler_core.workspace.Workspace):
    def create_or_find_figure(self, title=None, subplots=None, window_name=None):
        assert not self.__in_background

        if title is None:
            title = self.__module.module_name

        if window_name is None:
            window_name = cellprofiler.gui.figure.window_name(self.__module)

        if self.__create_new_window:
            figure = cellprofiler.gui.figure.Figure(
                parent=self, title=title, name=window_name, subplots=subplots
            )
        else:
            figure = cellprofiler.gui.figure.create_or_find(
                self.__frame, title=title, name=window_name, subplots=subplots
            )

        if figure not in self.__windows_used:
            self.__windows_used.append(figure)

        return figure

    def get_module_figure(self, module, image_set_number, parent=None):
        assert not self.__in_background

        window_name = cellprofiler.gui.figure.window_name(module)

        if self.measurements.has_feature(
            cellprofiler_core.measurement.EXPERIMENT,
            cellprofiler_core.measurement.M_GROUPING_TAGS,
        ):
            group_number = self.measurements[
                cellprofiler_core.measurement.IMAGE,
                cellprofiler_core.measurement.GROUP_NUMBER,
                image_set_number,
            ]

            group_index = self.measurements[
                cellprofiler_core.measurement.IMAGE,
                cellprofiler_core.measurement.GROUP_INDEX,
                image_set_number,
            ]

            title = "{} #{}, image cycle #{}, group #{}, group index #{}".format(
                module.module_name,
                module.module_num,
                image_set_number,
                group_number,
                group_index,
            )
        else:
            title = "{} #{}, image cycle #{}".format(
                module.module_name, module.module_num, image_set_number
            )

        if self.__create_new_window:
            figure = cellprofiler.gui.figure.Figure(
                parent=parent or self.__frame, name=window_name, title=title
            )
        else:
            figure = cellprofiler.gui.figure.create_or_find(
                parent=parent or self.__frame, name=window_name, title=title
            )
            figure.Title = title

        if figure not in self.__windows_used:
            self.__windows_used.append(figure)

        return figure
