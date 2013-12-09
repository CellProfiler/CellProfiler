'''<b>ClassifyPixels</b> classify image pixels as belonging to different 
classes using the machine-learning tool, ilastik.
<hr>

ClassifyPixels performs per-pixel classification using the 
<a href="http://www.ilastik.org/">ilastik</a> application.
Ilastik is now bundled with the CellProfiler distribution; it applies
supervised machine learning techniques to images to learn their features.
A user trains a classifier with Ilastik and then saves the classifier.
The user then uses the ClassifyPixels module to classify the pixels in an
image. 

ClassifyPixels produces an "image" consisting of probabilities that
the pixel belongs to the chosen class; this image is similar to
an intensity image that would be produced by fluorescence imaging.
Provided that the classifier is sufficiently accurate, the image is
well-suited for input into one of the <b>Identify</b> modules for
object detection. More instructions on using the interface may be found 
<a href="http://ilastik.org/index.php?cat=20_Documentation&page=03_Cellprofiler">here</a>.
Please note that you must use the same image format for classification
as for the initial learning phase.

Currently, ilastik is only available for Windows, and is accessible from
in the CellProfiler folder under the Start Menu. A 64-bit system is 
recommended for running ilastik.
'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

import urllib

import cellprofiler.cpmodule as cpm
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage  as cpi

from cellprofiler.preferences import standardize_default_folder_names, \
     DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, NO_FOLDER_NAME, \
     ABSOLUTE_FOLDER_NAME, IO_FOLDER_CHOICE_HELP_TEXT, \
     DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, \
     URL_FOLDER_NAME
  
import logging
logger = logging.getLogger(__name__)
import numpy as np
import sys, os

# Import vigra
try:
    import vigra
except ImportError, vigraImport:
    logger.warning("""vigra import: failed to import the vigra library. Please follow the instructions on 
"http://hci.iwr.uni-heidelberg.de/vigra/" to install vigra""", exc_info=True)
    raise vigraImport

# Import h5py
try:
    import h5py
except ImportError, h5pyImport:
    logger.warning("""h5py import: failed to import the h5py library.""", 
                   exc_info=True)
    raise h5pyImport
    
# Import ilastik 

old_stdout = sys.stdout
try:
    sys.stdout = sys.stderr = open(os.devnull, "w")
    from ilastik.core.dataMgr import DataMgr, DataItemImage
    from ilastik.modules.classification.core.featureMgr import FeatureMgr
    from ilastik.modules.classification.core.classificationMgr import ClassificationMgr
    from ilastik.modules.classification.core.features.featureBase import FeatureBase
    from ilastik.modules.classification.core.classifiers.classifierRandomForest import ClassifierRandomForest
    from ilastik.modules.classification.core.classificationMgr import ClassifierPredictThread
    from ilastik.core.volume import DataAccessor
    sys.stdout = old_stdout
    
except ImportError, ilastikImport:
    sys.stdout = old_stdout
    logger.warning("""ilastik import: failed to import the ilastik. Please follow the instructions on 
"http://www.ilastik.org" to install ilastik""", exc_info=True)
    raise ilastikImport

CLASSIFIERS_KEY = "IlastikClassifiers"
FEATURE_ITEMS_KEY = "IlastikFeatureItems"

SI_PROBABILITY_MAP_COUNT = 3

#
# Classifiers by file. Key is file name, value is a tuple of stat modification
# time and classifier
#
classifier_dict = {}

class ClassifyPixels(cpm.CPModule):
    module_name = 'ClassifyPixels'
    variable_revision_number = 2
    category = "Image Processing"
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image", cps.NONE)
        
        self.probability_maps = []
        
        self.probability_map_count = cps.HiddenCount(
            self.probability_maps, "Probability map count")
        
        self.add_probability_map(False)
        
        self.add_probability_button = cps.DoSomething(
            "Add another probability map", "Add", self.add_probability_map,doc = """
            Press the <i>Add</i> button to output another
            probability map image from the classifier. Ilastik can be trained
            to recognize any number of classes of pixels. You can generate
            probability maps for any or all of them simultaneously by adding
            more images.""")
        
        self.h5_directory = cps.DirectoryPath(
            "Classifier file location",
            dir_choices = [
                DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_INPUT_FOLDER_NAME, 
                ABSOLUTE_FOLDER_NAME, DEFAULT_INPUT_SUBFOLDER_NAME,
                DEFAULT_OUTPUT_SUBFOLDER_NAME, URL_FOLDER_NAME], 
            allow_metadata = False,doc ="""
                Select the folder containing the classifier file to be loaded. 
            %(IO_FOLDER_CHOICE_HELP_TEXT)s"""%globals())
        
        def get_directory_fn():
            '''Get the directory for the CSV file name'''
            return self.h5_directory.get_absolute_path()
        
        def set_directory_fn(path):
            dir_choice, custom_path = self.h5_directory.get_parts_from_path(path)
            self.h5_directory.join_parts(dir_choice, custom_path)
                
        self.classifier_file_name = cps.FilenameText(
            "Classfier file name",
            cps.NONE,
            doc="""This is the name of the Classfier file.""",
            get_directory_fn = get_directory_fn,
            set_directory_fn = set_directory_fn,
            browse_msg = "Choose Classifier file",
            exts = [("Classfier file (*.h5)","*.h5"),("All files (*.*)","*.*")]
        )
        
    def add_probability_map(self, can_remove=True):
        group = cps.SettingsGroup()
        group.can_remove = can_remove
        self.probability_maps.append(group)
        
        # The following settings are used for the combine option
        group.output_image = cps.ImageNameProvider(
            "Name the output probability map", "ProbabilityMap")
        
        group.class_sel = cps.Integer(
            "Select the class", 
            0, 0, 42, doc=
            '''Select the class you want to use. The class number 
            corresponds to the label-class in ilastik''')
        
        if can_remove:
            group.remover = cps.RemoveSettingButton(
                "Remove probability map", 
                "Remove", self.probability_maps, group,doc = """
                Press the <i>Remove</i> button to remove the
                probability map image from the list of images produced by this
                module""")
        
    def settings(self):
        result = [self.image_name, self.h5_directory, self.classifier_file_name,
                  self.probability_map_count]
        for group in self.probability_maps:
            result += [group.output_image, group.class_sel]
        return result
    
    def visible_settings(self):
        result = [self.image_name]
        for group in self.probability_maps:
            result += [group.output_image, group.class_sel]
            if group.can_remove:
                result += [group.remover]
        result += [self.add_probability_button, self.h5_directory,
                   self.classifier_file_name]
        return result
        
    def run(self, workspace):
        # get input image
        image = workspace.image_set.get_image(self.image_name.value, must_be_color=False) 
        
        # recover raw image domain
        image_ = image.pixel_data
        if image.get_scale() is not None:
            image_ = image_ * image.get_scale()
        else:
            # Best guess for derived images
            image_ = image_ * 255.0
        #
        # Apply a rescaling that's done similarly in ilastik's dataImpex
        #
        image_max = np.max(image_)
        if (image_max > 255) and (image_max < 4096):
            image_ = image_ / 4095. * 255.0
        
        # Create ilastik dataMgr
        dataMgr = DataMgr()
        
        # Transform input image to ilastik convention s
        # 3D = (time,x,y,z,channel) 
        # 2D = (time,1,x,y,channel)
        # Note, this work for 2D images right now. Is there a need for 3D
        image_.shape = (1,1) + image_.shape
        
        # Check if image_ has channels, if not add singelton dimension
        if len(image_.shape) == 4:
            image_.shape = image_.shape + (1,)
        
        # Add data item di to dataMgr
        di = DataItemImage('')
        di.setDataVol(DataAccessor(image_))
        dataMgr.append(di, alreadyLoaded=True)
        dataMgr.module["Classification"]["classificationMgr"].classifiers =\
            self.get_classifiers(workspace)

        # Create FeatureMgr
        fm = FeatureMgr(dataMgr, self.get_feature_items(workspace))
        
        # Compute features

        fm.prepareCompute(dataMgr)
        fm.triggerCompute()
        fm.joinCompute(dataMgr)
        
        # Predict with loaded classifier
        
        classificationPredict = ClassifierPredictThread(dataMgr)
        classificationPredict.start()
        classificationPredict.wait()
        
        workspace.display_data.source_image = image.pixel_data
        workspace.display_data.dest_images = []
        for group in self.probability_maps:
            # Produce output image and select the probability map
            probMap = classificationPredict._prediction[0][
                0 ,0 ,: ,: ,int(group.class_sel.value)]
            temp_image = cpi.Image(probMap, parent_image=image)
            workspace.image_set.add(group.output_image.value, temp_image)
            workspace.display_data.dest_images.append(probMap)

    def get_classifiers(self, workspace):
        d = self.parse_classifier_file(workspace)
        return d[CLASSIFIERS_KEY]
    
    def get_feature_items(self, workspace):
        d = self.parse_classifier_file(workspace)
        return d[FEATURE_ITEMS_KEY]
        
    def parse_classifier_file(self, workspace):
        global classifier_dict
        # Load classifier from hdf5
        if self.h5_directory.dir_choice == URL_FOLDER_NAME:
            url = self.classifier_file_name.value.encode("utf-8")
            if url in classifier_dict:
                last_modtime, d = classifier_dict[url]
                return d
            filename, headers = urllib.urlretrieve(url)
            try:
                modtime = os.stat(filename).st_mtime
                d = self.parse_classifier_hdf5(filename)
                classifier_dict[url] = (modtime, d)
            finally:
                os.remove(filename)
        else:
            fileName = os.path.join(
                self.h5_directory.get_absolute_path(), 
                self.classifier_file_name.value).encode("utf-8")
            modtime = os.stat(fileName).st_mtime
            if fileName in classifier_dict:
                last_modtime, d = classifier_dict[fileName]
                if modtime == last_modtime:
                    return d
            d = self.parse_classifier_hdf5(fileName)
            classifier_dict[fileName] = (modtime, d)
        return d
    
    def parse_classifier_hdf5(self, filename):
        '''Parse the classifiers out of the HDF5 file
        
        filename - name of classifier file
        
        returns a dictionary
           CLASSIFIERS_KEY - the random forest classifiers
           FEATURE_ITEMS_KEY - the features needed by the classifier
        '''
        d = {}
        if not isinstance(filename, str):
            filename = filename.encode('utf-8')
        hf = h5py.File(filename,'r')
        temp = hf['classifiers'].keys()
        # If hf is not closed this leads to an error in win64 and mac os x
        hf.close()
        del hf
        
        classifiers = []
        for cid in temp:
            cidpath = 'classifiers/' + cid
            try:
                classifiers.append(
                    ClassifierRandomForest.deserialize(filename, cidpath))
            except:
                classifiers.append(
                    ClassifierRandomForest.loadRFfromFile(filename, cidpath))

        d[CLASSIFIERS_KEY] = classifiers
        
        # Restore user selection of feature items from hdf5
        featureItems = []
        f = h5py.File(filename,'r')
        for fgrp in f['features'].values():
            featureItems.append(FeatureBase.deserialize(fgrp))
        d[FEATURE_ITEMS_KEY] = featureItems
        f.close()
        del f
        return d

    def display(self, workspace, figure):
        figure.set_subplots((len(workspace.display_data.dest_images) + 1, 1))
        source_image = workspace.display_data.source_image
        if source_image.ndim == 3:
            src_plot = figure.subplot_imshow_color(
                0, 0, source_image, title = self.image_name.value)
        else:
            src_plot = figure.subplot_imshow_grayscale(
                0, 0, source_image, title = self.image_name.value)
        for i, dest_image in enumerate(workspace.display_data.dest_images):
            figure.subplot_imshow_grayscale(
                i + 1, 0, dest_image,
                title = self.probability_maps[i].output_image.value,
                sharexy = src_plot)

    def prepare_settings(self, setting_values):
        '''Prepare the module to receive the settings'''
        n_maps = int(setting_values[SI_PROBABILITY_MAP_COUNT])
        if len(self.probability_maps) > n_maps:
            del self.probability_maps[n_maps:]
        elif len(self.probability_maps) < n_maps:
            for _ in range(len(self.probability_maps), n_maps):
                self.add_probability_map()

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        '''Upgrade settings to maintain backwards compatibility
        
        setting_values - list of setting strings
        variable_revision_number - version number used to save the settings
        module_name - original module name used to save the settings
        from_matlab - true if CellProfiler 1.0 pipeline
        '''
        if variable_revision_number == 1:
            setting_values = [
                setting_values[0], # image_name
                setting_values[3], # h5_directory
                setting_values[4], # classifier_file_name
                "1",               # probability map count = 1
                setting_values[1], # output_image
                setting_values[2]] # class_sel
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

