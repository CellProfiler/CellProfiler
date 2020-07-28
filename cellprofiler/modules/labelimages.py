"""
LabelImages
===========

**LabelImages** assigns plate metadata to image sets.

**LabelImages** assigns a plate number, well and site number to each
image set based on the order in which they are processed. You can use
**Label Images** to add plate and well metadata for images loaded using
*Order* for “Image set matching order” in **NamesAndTypes**.

LabelImages assumes the following are true of the image order:

-  Each well has the same number of images (i.e., sites) per channel.
-  Each plate has the same number of rows and columns, so that the total
   number of images per plate is the same.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also the **Metadata** module.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Metadata_Plate:* The plate number, starting at 1 for the first
   plate.
-  *Metadata_Well:* The well name, e.g., *A01*.
-  *Metadata_Row:* The row name, starting with *A* for the first row.
-  *Metadata_Column:* The column number, starting with 1 for the first
   column.
-  *Metadata_Site:* The site number within the well, starting at 1 for
   the first site.

"""

from functools import reduce

import numpy
from cellprofiler_core.constants.measurement import COLTYPE_INTEGER
from cellprofiler_core.constants.measurement import COLTYPE_VARCHAR_FORMAT
from cellprofiler_core.constants.measurement import C_METADATA
from cellprofiler_core.constants.measurement import FTR_COLUMN
from cellprofiler_core.constants.measurement import FTR_PLATE
from cellprofiler_core.constants.measurement import FTR_ROW
from cellprofiler_core.constants.measurement import FTR_SITE
from cellprofiler_core.constants.measurement import FTR_WELL
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.constants.measurement import M_COLUMN
from cellprofiler_core.constants.measurement import M_PLATE
from cellprofiler_core.constants.measurement import M_ROW
from cellprofiler_core.constants.measurement import M_SITE
from cellprofiler_core.constants.measurement import M_WELL
from cellprofiler_core.module import Module
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.text.number import Integer

O_ROW = "Row"
O_COLUMN = "Column"


class LabelImages(Module):
    module_name = "LabelImages"
    category = "File Processing"
    variable_revision_number = 1

    def create_settings(self):
        self.site_count = Integer(
            "Number of image sites per well",
            1,
            minval=1,
            doc="""\
Enter the number of image sets (fields of view) corresponding to each well.""",
        )

        self.column_count = Integer(
            "Number of columns per plate",
            12,
            minval=1,
            doc="""\
Enter the number of columns per plate.""",
        )

        self.row_count = Integer(
            "Number of rows per plate",
            8,
            minval=1,
            doc="""\
Enter the number of rows per plate.""",
        )

        self.order = Choice(
            "Order of image data",
            [O_ROW, O_COLUMN],
            doc="""\
This setting specifies how the input data is ordered (assuming that
sites within a well are ordered consecutively):

-  *%(O_ROW)s:* The data appears by row and then by column. That is,
   all columns for a given row (e.g., A01, A02, A03…) appear
   consecutively, for each row in consecutive order.
-  *%(O_COLUMN)s:* The data appears by column and then by row. That is,
   all rows for a given column (e.g., A01, B01, C01…) appear
   consecutively, for each column in consecutive order.

For instance, the SBS Bioimage example (available `here`_) has files that are named:
Channel1-01-A01.tif, Channel1-02-A02.tif, …, Channel1-12-A12.tif, Channel1-13-B01.tif, …
You would use “%(O_ROW)s” to label these because the ordering is by row and then by column.

.. _here: http://cellprofiler.org/examples.html#SBS_Bioimage_CNT
"""
            % globals(),
        )

    def settings(self):
        """The settings as they appear in the pipeline"""
        return [self.site_count, self.column_count, self.row_count, self.order]

    def run(self, workspace):
        """Run one image set"""
        m = workspace.measurements
        well_count, site_index = divmod(m.image_set_number - 1, self.site_count.value)
        if self.order == O_ROW:
            row_count, column_index = divmod(well_count, self.column_count.value)
            plate_index, row_index = divmod(row_count, self.row_count.value)
        else:
            column_count, row_index = divmod(well_count, self.row_count.value)
            plate_index, column_index = divmod(column_count, self.column_count.value)

        row_text_indexes = [
            x % 26
            for x in reversed(
                [int(row_index / (26 ** i)) for i in range(self.row_digits)]
            )
        ]

        row_text = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[x] for x in row_text_indexes]
        row_text = reduce(lambda x, y: x + y, row_text)
        well_template = "%s%0" + str(self.column_digits) + "d"
        well = well_template % (row_text, column_index + 1)

        statistics = [
            (M_SITE, site_index + 1),
            (M_ROW, row_text),
            (M_COLUMN, column_index + 1),
            (M_WELL, well),
            (M_PLATE, plate_index + 1),
        ]
        for feature, value in statistics:
            m.add_image_measurement(feature, value)
        workspace.display_data.col_labels = ("Metadata", "Value")
        workspace.display_data.statistics = [
            (feature, str(value)) for feature, value in statistics
        ]

    @property
    def row_digits(self):
        """The number of letters it takes to represent a row.

        If a plate has more than 26 rows, you need two digits. The following
        is sufficiently general.
        """
        return int(1 + numpy.log(self.row_count.value) / numpy.log(26))

    @property
    def column_digits(self):
        """The number of digits it takes to represent a column."""

        return int(1 + numpy.log10(self.column_count.value))

    def get_measurement_columns(self, pipeline):
        row_coltype = COLTYPE_VARCHAR_FORMAT % self.row_digits
        well_coltype = COLTYPE_VARCHAR_FORMAT % (self.row_digits + self.column_digits)
        return [
            (IMAGE, M_SITE, COLTYPE_INTEGER),
            (IMAGE, M_ROW, row_coltype),
            (IMAGE, M_COLUMN, COLTYPE_INTEGER),
            (IMAGE, M_WELL, well_coltype),
            (IMAGE, M_PLATE, COLTYPE_INTEGER),
        ]

    def get_categories(self, pipeline, object_name):
        if object_name == IMAGE:
            return [C_METADATA]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == IMAGE and category == C_METADATA:
            return [
                FTR_SITE,
                FTR_ROW,
                FTR_COLUMN,
                FTR_WELL,
                FTR_PLATE,
            ]
        return []

    def display(self, workspace, figure):
        """Display the plate / well information in a figure table"""
        figure.set_subplots((1, 1))
        figure.subplot_table(
            0,
            0,
            workspace.display_data.statistics,
            col_labels=workspace.display_data.col_labels,
        )
