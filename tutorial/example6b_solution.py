'''<b>Example6b</b> Creating an image set list
<hr>
'''
import datetime
import os

from bioformats import load_using_bioformats

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.pipeline as cpp
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

M_FIRST_TIME = "Example6b_FirstTime"

class Example6b(cpm.CPModule):
    module_name = "Example6b"
    variable_revision_number = 1
    category = "File Processing"
    
    def create_settings(self):
        self.image_name = cps.ImageNameProvider(
            "Input image name", "TodaysImage")
        #
        # The DirectoryPath setting lets the user pick a directory
        # to be scanned.
        #
        self.folder = cps.DirectoryPath("Folder")
        
    def settings(self):
        return [self.image_name, self.folder]
    
    def is_load_module(self):
        '''Tell CellProfiler that this module produces image sets'''
        return True
    
    def prepare_run(self, workspace):
        measurements = workspace.measurements
        image_name = self.image_name.value
        yesterday = datetime.datetime.utcnow() - datetime.timedelta(1)
        iso_yesterday = yesterday.isoformat() 
        M_FILE_NAME = cpmeas.C_FILE_NAME + "_" + image_name
        M_PATH_NAME = cpmeas.C_PATH_NAME + "_" + image_name
        #
        # call measurements.add_experiment_measurement to add iso_yesterday
        # as M_FIRST_TIME
        #
        measurements.add_experiment_measurement(M_FIRST_TIME, iso_yesterday)
        for i, (pathname, filename) in enumerate(self.get_files(yesterday)):
            image_number = i+1
            #
            # Save the pathname in the M_PATH_NAME measurement
            # Save the filename in the M_FILE_NAME measurement
            #
            # measurements.add_measurement has an optional "image_set_number"
            # argument which you should use to store each file with the
            # correct image number
            # 
            measurements.add_measurement(cpmeas.IMAGE, M_PATH_NAME, pathname,
                                         image_set_number = image_number)
            measurements.add_measurement(cpmeas.IMAGE, M_FILE_NAME, filename,
                                         image_set_number = image_number)
        return True
    
    def run(self, workspace):
        image_name = self.image_name.value
        measurements = workspace.measurements
        image_set = workspace.image_set
        
        M_FILE_NAME = cpmeas.C_FILE_NAME + "_" + image_name
        M_PATH_NAME = cpmeas.C_PATH_NAME + "_" + image_name
        #
        # Use measurements.get_measurement to get the file name and path name
        # for this cycle
        #
        path_name = measurements.get_measurement(cpmeas.IMAGE, M_PATH_NAME)
        file_name = measurements.get_measurement(cpmeas.IMAGE, M_FILE_NAME)
        #
        # use os.path.join(pathname, filename) to get the name for BioFormats
        #
        path = os.path.join(path_name, file_name)
        #
        # call load_using_bioformats(path) to load the file into a numpy array
        #
        pixel_data = load_using_bioformats(path)
        #
        # make a cpi.Image for the image data
        #
        image = cpi.Image(pixel_data, path_name = path_name, file_name = file_name)
        #
        # call image_set.add to add the image to the image set
        #
        image_set.add(image_name, image)
        
    def get_files(self, utctime):
        '''Return a list of path / filename tuples created after utctime
        
        utctime - a utc datetime representing the earliest acceptable creation time
        
        returns a list of two-tuples with the path in the first tuple and
                the file in the second.
        '''
        result = []
        for root, dirnames, filenames in os.walk(self.folder.get_absolute_path()):
            for filename in filenames:
                if not filename.lower().endswith(".tif"):
                    continue
                path = os.path.join(root, filename)
                t = datetime.datetime.utcfromtimestamp(os.stat(path).st_ctime)
                if t >= utctime:
                    result.append((root, filename))
        return result