# coding=utf-8

"""
DisplayPlatemap
===============

**DisplayPlatemap** displays a desired measurement in a plate map view.

**DisplayPlatemap** is a tool for browsing image-based data laid out on
multi-well plates common to high-throughput biological screens. The
display window for this module shows a plate map with each well
color-coded according to the measurement chosen.

As the pipeline runs, the measurement information displayed is updated,
so the value shown for each well is current up to the image cycle
currently being processed; wells that have no corresponding
measurements as yet are shown as blank.

At this time, the display produced when **DisplayPlatemap** is run as a
module cannot be saved in the pipeline (e.g., by using **SaveImages**). The
display can be saved manually by selecting the window produced by the
module and clicking the Save icon in its menu bar or by choosing *File
> Save* from CellProfiler's main menu bar.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also other **Display** modules and data tools.
"""

import numpy as np

import six

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps
from cellprofiler.modules._help import USING_METADATA_HELP_REF

AGG_AVG = 'avg'
AGG_MEDIAN = 'median'
AGG_STDEV = 'stdev'
AGG_CV = 'cv%'
AGG_NAMES = [AGG_AVG, AGG_STDEV, AGG_MEDIAN, AGG_CV]
OI_OBJECTS = 'Object'
OI_IMAGE = 'Image'
WF_NAME = 'Well name'
WF_ROWCOL = 'Row & Column'


class DisplayPlatemap(cpm.Module):
    module_name = "DisplayPlatemap"
    category = "Data Tools"
    variable_revision_number = 2

    def get_object(self):
        if self.objects_or_image.value == OI_OBJECTS:
            return self.object.value
        else:
            return cpmeas.IMAGE

    def create_settings(self):
        self.objects_or_image = cps.Choice(
                "Display object or image measurements?",
                [OI_OBJECTS, OI_IMAGE], doc="""\
-  *%(OI_IMAGE)s* allows you to select an image measurement to display
   for each well.
-  *%(OI_OBJECTS)s* allows you to select an object measurement to
   display for each well.
""" % globals())

        self.object = cps.ObjectNameSubscriber(
                'Select the object whose measurements will be displayed',
                cps.NONE, doc='''\
Choose the name of objects identified by some previous module (such as
**IdentifyPrimaryObjects** or **IdentifySecondaryObjects**)
whose measurements are to be displayed.
''')

        self.plot_measurement = cps.Measurement(
                'Select the measurement to plot',
                self.get_object, cps.NONE, doc='''Choose the image or object measurement made by a previous module to plot.''')

        self.plate_name = cps.Measurement('Select your plate metadata',
                                          lambda: cpmeas.IMAGE,
                                          'Metadata_Plate', doc='''\
Choose the metadata tag that corresponds to the plate identifier. That
is, each plate should have a metadata tag containing a specifier
corresponding uniquely to that plate.

%(USING_METADATA_HELP_REF)s
''' % globals())

        self.plate_type = cps.Choice(
                'Multiwell plate format',
                ['96', '384'], doc='''\
The module assumes that your data is laid out in a multi-well plate
format common to high-throughput biological screens. Supported formats
are:

-  *96:* A 96-well plate with 8 rows × 12 columns
-  *384:* A 384-well plate with 16 rows × 24 columns
''')

        self.well_format = cps.Choice(
                "Well metadata format",
                [WF_NAME, WF_ROWCOL], doc="""\
-  *%(WF_NAME)s* allows you to select an image measurement to display
   for each well.
-  *%(WF_ROWCOL)s* allows you to select an object measurement to
   display for each well.
""" % globals())

        self.well_name = cps.Measurement(
                'Select your well metadata',
                lambda: cpmeas.IMAGE, 'Metadata_Well', doc='''\
Choose the metadata tag that corresponds to the well identifier. The
row-column format of these entries should be an alphabetical character
(specifying the plate row), followed by two integer characters
(specifying the plate column). For example, a standard format 96-well
plate would span from “A1” to “H12”, whereas a 384-well plate (16 rows
and 24 columns) would span from well “A01” to well “P24”."

%(USING_METADATA_HELP_REF)s
''' % globals())

        self.well_row = cps.Measurement('Select your well row metadata',
                                        lambda: cpmeas.IMAGE, 'Metadata_WellRow', doc='''\
Choose the metadata tag that corresponds to the well row identifier,
typically specified as an alphabetical character. For example, a
standard format 96-well plate would span from row “A” to “H”, whereas a
384-well plate (16 rows and 24 columns) would span from row “A” to “P”.

%(USING_METADATA_HELP_REF)s
''' % globals())

        self.well_col = cps.Measurement('Select your well column metadata',
                                        lambda: cpmeas.IMAGE, 'Metadata_WellCol', doc='''\
Choose the metadata tag that corresponds to the well column identifier,
typically specified with two integer characters. For example, a standard
format 96-well plate would span from column “01” to “12”, whereas a
384-well plate (16 rows and 24 columns) would span from column “01” to
“24”.

%(USING_METADATA_HELP_REF)s
''' % globals())

        self.agg_method = cps.Choice(
                'How should the values be aggregated?',
                AGG_NAMES, AGG_NAMES[0], doc='''\
Measurements must be aggregated to a single number for each well so that
they can be represented by a color. Options are:

-  *%(AGG_AVG)s:* Average
-  *%(AGG_STDEV)s:* Standard deviation
-  *%(AGG_MEDIAN)s*
-  *%(AGG_CV)s:* Coefficient of variation, defined as the ratio of the
   standard deviation to the mean. This is useful for comparing between
   data sets with different units or widely different means.
''' % globals())

        self.title = cps.Text(
                'Enter a title for the plot, if desired', '',
                doc='''\
Enter a title for the plot. If you leave this blank, the title will
default to *(cycle N)* where *N* is the current image cycle being
executed.
''')

    def settings(self):
        return [self.objects_or_image, self.object, self.plot_measurement,
                self.plate_name, self.plate_type, self.well_name,
                self.well_row, self.well_col, self.agg_method, self.title,
                self.well_format]

    def visible_settings(self):
        result = [self.objects_or_image]
        if self.objects_or_image.value == OI_OBJECTS:
            result += [self.object]
        result += [self.plot_measurement]
        result += [self.plate_type]
        result += [self.plate_name]
        result += [self.well_format]
        if self.well_format == WF_NAME:
            result += [self.well_name]
        elif self.well_format == WF_ROWCOL:
            result += [self.well_row, self.well_col]
        result += [self.agg_method, self.title]
        return result

    def run(self, workspace):
        if self.show_window:
            m = workspace.get_measurements()
            # Get plates
            plates = map(
                    six.text_type,
                    m.get_all_measurements(cpmeas.IMAGE, self.plate_name.value))
            # Get wells
            if self.well_format == WF_NAME:
                wells = m.get_all_measurements(cpmeas.IMAGE, self.well_name.value)
            elif self.well_format == WF_ROWCOL:
                wells = ['%s%s' % (x, y) for x, y in zip(m.get_all_measurements(cpmeas.IMAGE, self.well_row.value),
                                                         m.get_all_measurements(cpmeas.IMAGE, self.well_col.value))]
            # Get data to plot
            data = m.get_all_measurements(self.get_object(), self.plot_measurement.value)

            # Construct a dict mapping plates and wells to lists of measurements
            pm_dict = {}
            for plate, well, data in zip(plates, wells, data):
                if data is None:
                    continue
                if plate in pm_dict:
                    if well in pm_dict[plate]:
                        pm_dict[plate][well] += [data]
                    else:
                        pm_dict[plate].update({well: [data]})
                else:
                    pm_dict[plate] = {well: [data]}

            for plate, sub_dict in pm_dict.items():
                for well, vals in sub_dict.items():
                    vals = np.hstack(vals)
                    if self.agg_method == AGG_AVG:
                        pm_dict[plate][well] = np.mean(vals)
                    elif self.agg_method == AGG_STDEV:
                        pm_dict[plate][well] = np.std(vals)
                    elif self.agg_method == AGG_MEDIAN:
                        pm_dict[plate][well] = np.median(vals)
                    elif self.agg_method == AGG_CV:
                        pm_dict[plate][well] = np.std(vals) / np.mean(vals)
                    else:
                        raise NotImplemented
            workspace.display_data.pm_dict = pm_dict

    def display(self, workspace, figure):
        pm_dict = workspace.display_data.pm_dict
        if not hasattr(figure, "subplots"):
            figure.set_subplots((1, 1))
        if self.title.value != '':
            title = '%s (cycle %s)' % (self.title.value, workspace.measurements.image_set_number)
        else:
            title = '%s(%s)' % (self.agg_method, self.plot_measurement.value)
        figure.subplot_platemap(0, 0, pm_dict, self.plate_type,
                                title=title)

    def run_as_data_tool(self, workspace):
        return self.run(workspace)

    def backwards_compatibilize(self, setting_values, variable_revision_number,
                                module_name, from_matlab):
        if variable_revision_number == 1:
            # Add the wellformat setting
            setting_values += [WF_NAME]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
