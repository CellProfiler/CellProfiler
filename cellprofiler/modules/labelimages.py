'''<b>LabelImages</b> assigns plate metadata to image sets.
<hr>

<b>LabelImages</b> assigns a plate number, well and site number to each image
set based on the order in which they are processed. You can use <b>Label
Images</b> to add plate and well metadata for images loaded using
<b>LoadImages</b>, using the file order instead of the text in the file
name. <b>LabelImages</b> assumes that you have an identical number of
image sets per well (the <i>sites</b>), and an identical number of rows and 
columns of wells per plate with complete file lists for all but the last
plate and well.
Files have to be loaded in one of the two following orders to use
<b>LabelImages</b><br>
<ul>
<lh>By row</ul>
<li>All sites for a given well appear consecutively</li>
<li>All wells for a given row (e.g. A01, A02, A03...) appear consecutively</li>
<li>All rows for a given column appear consecutively</li>
<li>All columns for a given plate appear consecutively</li>
</ul>
<br>
or
<br>
<ul>
<lh>By column</ul>
<li>All sites for a given well appear consecutively</li>
<li>All wells for a given column (e.g. A01, B01, C01...) appear consecutively</li>
<li>All columns for a row column appear consecutively</li>
<li>All rows for a given plate appear consecutively</li>
</ul>
<p>
<b>LabelImages</b> adds the following measurements to the image table:
<br>
<table>
<tr><th>Measurement</th><th>Description</th></tr>
<tr><td>Metadata_Plate</td><td>The plate number, starting at 1 for the first plate</td></tr>
<tr><td>Metadata_Well</td><td>The well name, for instance, "A01"</td></tr>
<tr><td>Metadata_Row</td><td>The row name, for instance, "A"</td></tr>
<tr><td>Metadata_Column</td><td>The column number</td></tr>
<tr><td>Metadata_Site</td><td>The site number within the well</td></tr>
</table>
'''
__version__ = "$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

O_ROW = "Row"
O_COLUMN = "Column"

class LabelImages(cpm.CPModule):
    module_name = "LabelImages"
    category = "Other"
    variable_revision_number = 1
    
    def create_settings(self):
        self.site_count = cps.Integer(
            "# sites / well:", 1, minval=1,
            doc = """This setting controls the number of image sets for each well""")
        self.column_count = cps.Integer(
            "# of columns:", 12, minval=1,
            doc = """The number of columns per plate""")
        self.row_count = cps.Integer(
            "# of rows:", 8, minval=1,
            doc = """The number of rows per plate""")
        self.order = cps.Choice(
            "Order:", [O_ROW, O_COLUMN],
            doc = """This setting controls whether the data is ordered by
            row and then by column or by column and then by row. Choose, 
            "%(O_ROW)s", if data appears by row and then by column. Choose,
            "%(O_COLUMN)s", if data appears by column and then by row. For
            instance, the ExampleSBSImages sample has files that are named:<br>
            Channel1-01-A01.tif<br>
            Channel1-02-A02.tif<br>
            ...<br>
            Channel1-12-A12.tif<br>
            Channel1-13-B01.tif<br>
            ...<br>
            You would use "%(O_ROW)s" to label these because the ordering
            is by row and then by column.""" % globals())
        
    def settings(self):
        '''The settings as they appear in the pipeline'''
        return [self.site_count, self.column_count, self.row_count,
                self.order]
    
    def is_interactive(self):
        return False
    
    def run(self, workspace):
        '''Run one image set'''
        m = workspace.measurements
        well_count, site_index = divmod(m.image_set_index, self.site_count.value)
        if self.order == O_ROW:
            row_count, column_index = divmod(well_count, self.column_count.value)
            plate_index, row_index = divmod(row_count, self.row_count.value)
        else:
            column_count, row_index = divmod(well_count, self.row_count.value)
            plate_index, column_index = divmod(column_count, 
                                               self.column_count.value)
        
        row_text_indexes = [ 
            x % 26 for x in reversed(
                [int(row_index / (26 ** i)) for i in range(self.row_digits)])]
                             
        row_text = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ'[x] for x in row_text_indexes]
        row_text = reduce(lambda x,y: x+y, row_text)
        well_template = "%s%0" + str(self.column_digits) + "d"
        well = well_template % (row_text, column_index+1)
        
        statistics = [(cpmeas.M_SITE, site_index + 1),
                       (cpmeas.M_ROW, row_text),
                       (cpmeas.M_COLUMN, column_index + 1),
                       (cpmeas.M_WELL, well),
                       (cpmeas.M_PLATE, plate_index + 1)]
        for feature, value in statistics:
            m.add_image_measurement(feature, value)
        workspace.display_data.statistics = [("Metadata", "Value")]
        workspace.display_data.statistics += [
            (feature, str(value)) for feature, value in statistics]
    
    @property
    def row_digits(self):
        '''The number of letters it takes to represent a row.
        
        If a plate has more than 26 rows, you need two digits. The following
        is sufficiently general.
        '''
        return int(1 + np.log(self.row_count.value) / np.log(26))
        
    @property
    def column_digits(self):
        '''The number of digits it takes to represent a column.'''
        
        return int(1 + np.log10(self.column_count.value))
    
    def get_measurement_columns(self, pipeline):
        row_coltype = cpmeas.COLTYPE_VARCHAR_FORMAT % self.row_digits
        well_coltype = cpmeas.COLTYPE_VARCHAR_FORMAT % (
            self.row_digits + self.column_digits)
        return [
            (cpmeas.IMAGE, cpmeas.M_SITE, cpmeas.COLTYPE_INTEGER),
            (cpmeas.IMAGE, cpmeas.M_ROW, row_coltype),
            (cpmeas.IMAGE, cpmeas.M_COLUMN, cpmeas.COLTYPE_INTEGER),
            (cpmeas.IMAGE, cpmeas.M_WELL, well_coltype),
            (cpmeas.IMAGE, cpmeas.M_PLATE, cpmeas.COLTYPE_INTEGER)]
    
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [ cpmeas.C_METADATA ]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == cpmeas.C_METADATA:
            return [cpmeas.FTR_SITE, cpmeas.FTR_ROW, cpmeas.FTR_COLUMN,
                    cpmeas.FTR_WELL, cpmeas.FTR_PLATE]
        return []
    
    def display(self, workspace):
        '''Display the plate / well information in a figure table'''
        figure = workspace.create_or_find_figure(subplots=(1,1))
        figure.subplot_table(0, 0, workspace.display_data.statistics)
        
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        '''Upgrade the pipeline settings to the current revision of the module
        
        setting_values - setting strings from the pipeline
        variable_revision_number - revision of the module that saved the settings
        module_name - name of the module that saved the settings
        from_matlab - settings are from the Matlab version of CellProfiler
        '''
        if from_matlab and variable_revision_number == 1:
            label_name, rows_cols, row_or_column, image_cycles_per_well = \
                      setting_values
            row_count, column_count = rows_cols.split(',')
            if rows_cols == 'A02':
                order = O_ROW
            else:
                order = O_COLUMN
            setting_values = [ image_cycles_per_well, column_count, row_count,
                               order]
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab
