from cellprofiler_core import workspace
from cellprofiler_core.constants.measurement import EXPERIMENT
from cellprofiler_core.constants.measurement import GROUP_INDEX
from cellprofiler_core.constants.measurement import GROUP_NUMBER
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.constants.measurement import M_GROUPING_TAGS

import cellprofiler.gui.figure


class Workspace(workspace.Workspace):
    def create_or_find_figure(self, title=None, subplots=None, window_name=None):
        assert not self.__in_background

        if title is None:
            title = self.__module.module_name

        if window_name is None:
            from .utilities.figure import window_name

            window_name = window_name(self.__module)

        if self.__create_new_window:
            figure = cellprofiler.gui.figure.Figure(
                parent=self, title=title, name=window_name, subplots=subplots
            )
        else:
            from .utilities.figure import create_or_find

            figure = create_or_find(
                self.__frame, title=title, name=window_name, subplots=subplots
            )

        if figure not in self.__windows_used:
            self.__windows_used.append(figure)

        return figure

    def get_module_figure(self, module, image_set_number, parent=None):
        assert not self.__in_background

        from .utilities.figure import window_name

        window_name = window_name(module)

        if self.measurements.has_feature(EXPERIMENT, M_GROUPING_TAGS,):
            group_number = self.measurements[
                IMAGE, GROUP_NUMBER, image_set_number,
            ]

            group_index = self.measurements[
                IMAGE, GROUP_INDEX, image_set_number,
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
            from .utilities.figure import create_or_find

            figure = create_or_find(
                parent=parent or self.__frame, name=window_name, title=title
            )

            figure.Title = title

        if figure not in self.__windows_used:
            self.__windows_used.append(figure)

        return figure
