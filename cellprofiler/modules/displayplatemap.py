#TODO:
'''<b>Display Platemap </b> displays a desired measurement in plate map view
<hr>

A plate map is a...
'''

#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np
from contrib.english import ordinal

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas

AGG_AVG = 'avg'
AGG_MEDIAN = 'median'
AGG_STDEV = 'stdev'
AGG_CV = 'cv%'
AGG_NAMES = [AGG_AVG, AGG_STDEV, AGG_MEDIAN, AGG_CV]
OI_OBJECTS = 'Object'
OI_IMAGE = 'Image'
WF_NAME = 'Well name'
WF_ROWCOL = 'Row & Column'

class DisplayPlatemap(cpm.CPModule):
    
    module_name = "DisplayPlatemap"
    category = "Data Tools"
    variable_revision_number = 1
    
    def get_object(self):
        if self.objects_or_image.value == OI_OBJECTS:
            return self.object.value
        else:
            return cpmeas.IMAGE
    
    def create_settings(self):        
        self.objects_or_image = cps.Choice(
            "Display object or image measurements?",
            [OI_OBJECTS, OI_IMAGE],
            doc = """<ul><li> <i>%s</i> allows you to select an image 
            measurement to display for each well.</li> 
            <li><i>%s</i> allows you to select an object measurement to display 
            for each well.</li></ul>"""%(OI_IMAGE, OI_OBJECTS))
        
        self.object = cps.ObjectNameSubscriber(
            'Select the object whose measurements will be displayed','None',
            doc='''
            Choose the name of objects identified by some previous 
            module (such as <b>IdentifyPrimaryObjects</b> or 
            <b>IdentifySecondaryObjects</b>) whose measurements are to be displayed.''')
        
        self.plot_measurement = cps.Measurement(
            'Select the object measurement to plot', self.get_object, 'None',
            doc='''
            Choose the object measurement made by a previous 
            module to plot.''')
        
        self.plate_keys = []
        self.add_plate_measurement(False)
        self.add_plate_key = cps.DoSomething("","Add another plate identifier",
                                               self.add_plate)
        
        self.plate_type = cps.Choice(
            'What type of plate is the data from?',
            ['96','384'],
            doc = ''' ''')
        
        self.well_format = cps.Choice(
            "What form is your well metadata in?",
            [WF_NAME, WF_ROWCOL],
            doc = """<ul><li> <i>%s</i> allows you to select an image 
            measurement to display for each well.</li> 
            <li><i>%s</i> allows you to select an object measurement to display 
            for each well.</li></ul>"""%(WF_NAME, WF_ROWCOL))

        self.well_name = cps.Measurement('Select your well metadata', 
                                         lambda:cpmeas.IMAGE, 'Metadata_Well')

        self.well_row = cps.Measurement('Select your well row metadata', 
                                         lambda:cpmeas.IMAGE, 'Metadata_WellRow')
        
        self.well_col = cps.Measurement('Select your well column metadata', 
                                         lambda:cpmeas.IMAGE, 'Metadata_WellCol')

        self.agg_method = cps.Choice(
            'How should the values be aggregated?', 
            AGG_NAMES, AGG_NAMES[0],
            doc=''' ''')

        self.title = cps.Text(
            'Enter a title for the plot, if desired', '',
            doc = '''
            Enter a title for the plot. If you leave this blank,
            the title will default 
            to <i>(cycle N)</i> where <i>N</i> is the current image 
            cycle being executed.''')

    def add_plate_measurement(self, removable=True):
        # The text for these settings will be replaced in renumber_settings()
        group = cps.SettingsGroup()
        group.append('plate_meas', cps.Measurement('', lambda:cpmeas.IMAGE, 'Metadata_Plate'))
        if removable:
            group.append('remover', cps.RemoveSettingButton('', 
                        'Remove this plate identifier', self.plate_keys, group))
        self.plate_keys.append(group)
        
    def add_plate(self):
        self.add_plate_measurement()
        
    def renumber_settings(self):
        for idx, meas in enumerate(self.plate_keys):
            meas.plate_meas.text = 'Select the %s plate identifier'%(ordinal(idx + 1))
        
    def settings(self):
        result = [self.objects_or_image, self.object, self.plot_measurement]
        for group in self.plate_keys:
            result += group.settings
        result += [self.plate_type, self.well_name, self.well_row, 
                   self.well_col, self.agg_method, self.title]
        return result

    def visible_settings(self):
        self.renumber_settings()
        result = [self.objects_or_image]
        if self.objects_or_image.value == OI_OBJECTS:
            result += [self.object]
        result += [self.plot_measurement]
        for group in self.plate_keys:
            result += group.visible_settings()
        result += [self.plate_type]
        if self.well_format == WF_NAME:
            result += [self.well_name]
        elif self.well_format == WF_ROWCOL:
            result += [self.well_row, self.well_col]
        result += [self.agg_method, self.title]
        return result
        
    def run(self, workspace):
        if workspace.frame:
            m = workspace.get_measurements()
            # Get plates
            for plate in self.plate_keys:
                plates = m.get_all_measurements(cpmeas.IMAGE, plate.plate_meas.value)
            # Get wells
            if self.well_format == WF_NAME:
                wells = m.get_all_measurements(cpmeas.IMAGE, self.well_name.value)
            elif self.well_format == WF_ROWCOL:
                wells = ['%s%s'%(x,y) for x,y in zip(m.get_all_measurements(cpmeas.IMAGE, self.well_row.value),
                                                     m.get_all_measurements(cpmeas.IMAGE, self.well_col.value))]
            # Get data to plot
            data = m.get_all_measurements(self.get_object(), self.plot_measurement.value)

            # Construct a dict mapping plates and wells to lists of measurements
            pm_dict = {}
            for plate, well, data in zip(plates, wells, data):
                if plate in pm_dict:
                    if well in pm_dict[plate]:
                        pm_dict[plate][well] += [data]
                    else:
                        pm_dict[plate].update({well : [data]})
                else:
                    pm_dict[plate] = {well : [data]}

            for plate, sub_dict in pm_dict.items():            
                for well, vals in sub_dict.items():
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
            
            figure = workspace.create_or_find_figure(
                         title='Display platemap #%d'%(self.module_num), 
                         subplots=(1,1))
            if self.title.value != '':
                title = '%s (cycle %s)'%(self.title.value, workspace.measurements.image_set_number)
            else:
                title = '%s(%s)'%(self.agg_method, self.plot_measurement.value)
            figure.subplot_platemap(0, 0, pm_dict, self.plate_type,
                                    title=title)
            
    def run_as_data_tool(self, workspace):
        return self.run(workspace)

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        return setting_values, variable_revision_number, from_matlab
        
    