'''<b>ClassifyPixels</b> classify pixels using Ilastik
<hr>

ClassifyPixels performs per-pixel classification using the Ilastik application.
Ilastik is now bundled with the CellProfiler distribution; it applies
supervised machine learning techniques to images to learn their features.
A user trains a classifier with Ilastik and then saves the classifier.
The user then uses the ClassifyPixels module to classify the pixels in an
image. ClassifyPixels produces an "image" consisting of probabilities that
the pixel belongs to the chosen class.
'''
import cellprofiler.cpmodule as cpm
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage  as cpi


from cellprofiler.preferences import standardize_default_folder_names, \
     DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, NO_FOLDER_NAME, \
     ABSOLUTE_FOLDER_NAME, IO_FOLDER_CHOICE_HELP_TEXT
     
import numpy as np
import sys, os

# Import vigra
try:
    import vigra
except ImportError, vigraImport:
    print """vigra import: failed to import the vigra library. Please follow the instructions on 
"http://hci.iwr.uni-heidelberg.de/vigra/" to install vigra"""
    import traceback
    traceback.print_exc()
    raise vigraImport

# Import h5py
try:
    import h5py
except ImportError, h5pyImport:
    print """h5py import: failed to import the h5py library."""
    import traceback
    traceback.print_exc()
    raise h5pyImport
    
# Import ilastik 

try:
    from ilastik.core.dataMgr import DataMgr, DataItemImage
    from ilastik.modules.classification.core.featureMgr import FeatureMgr
    from ilastik.modules.classification.core.classificationMgr import ClassificationMgr
    from ilastik.modules.classification.core.features.featureBase import FeatureBase
    from ilastik.modules.classification.core.classifiers.classifierRandomForest import ClassifierRandomForest
    from ilastik.modules.classification.core.classificationMgr import ClassifierPredictThread
    from ilastik.core.volume import DataAccessor
    
except ImportError, ilastikImport:
    print """ilastik import: failed to import the ilastik. Please follow the instructions on 
"http://www.ilastik.org" to install ilastik"""
    import traceback
    traceback.print_exc()
    raise ilastikImport

class ClassifyPixels(cpm.CPModule):
    module_name = 'ClassifyPixels'
    variable_revision_number = 1
    category = "Image Processing"
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber("Select the input image", "None")
        
        # The following settings are used for the combine option
        self.output_image = cps.ImageNameProvider("Name of the output probability map", "ProbabilityMap")
        self.class_sel = cps.Integer("Class to choose", 0, 0, 42, doc='''Select the class you want to use. The class number corresponds to the label-class in ilastik''')
        
        self.h5_directory = cps.DirectoryPath(
            "Input classifier file location", allow_metadata = False,
            doc ="""Location of the input classifier file""")
        
        def get_directory_fn():
            '''Get the directory for the CSV file name'''
            return self.h5_directory.get_absolute_path()
        
        def set_directory_fn(path):
            dir_choice, custom_path = self.h5_directory.get_parts_from_path(path)
            self.h5_directory.join_parts(dir_choice, custom_path)
                
        self.classifier_file_name = cps.FilenameText(
            "Classfier File",
            "None",
            doc="""Classfier File""",
            get_directory_fn = get_directory_fn,
            set_directory_fn = set_directory_fn,
            browse_msg = "Choose Classifier file",
            exts = [("Classfier file (*.h5)","*.h5"),("All files (*.*)","*.*")]
        )
        
        self.browse_csv_button = cps.DoSomething(
            "Press to browse file contents", "View...", self.browse_csv)
    
    def browse_csv(self):
        import wx
        from cellprofiler.gui import get_cp_icon
        try:
            fd = self.open_csv()
        except:
            wx.MessageBox("Could not read %s" %self.csv_path)
            return
        reader = csv.reader(fd)
        header = reader.next()
        frame = wx.Frame(wx.GetApp().frame, title=self.csv_path)
        sizer = wx.BoxSizer(wx.VERTICAL)
        frame.SetSizer(sizer)
        list_ctl = wx.ListCtrl(frame, style = wx.LC_REPORT)
        sizer.Add(list_ctl, 1, wx.EXPAND)
        for i, field in enumerate(header):
            list_ctl.InsertColumn(i, field)
        for line in reader:
            list_ctl.Append(line)
        frame.SetMinSize((640,480))
        frame.SetIcon(get_cp_icon())
        frame.Fit()
        frame.Show()
    
    def settings(self):
        return [self.image_name, self.output_image, self.class_sel, self.h5_directory, self.classifier_file_name]
    
    def run(self, workspace):
        # get input image
        image = workspace.image_set.get_image(self.image_name.value, must_be_color=False) 
        
        # recover raw image domain
        image_ = image.pixel_data * image.get_scale()
        #
        # Apply a rescaling that's done similarly in ilastik's dataImpex
        #
        image_max = np.max(image_)
        if (image_max > 255) and (image_max < 4096):
            image_ = image_ / 4095. * 255.0
        
        # Create ilastik dataMgr
        self.dataMgr = DataMgr()
        
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
        self.dataMgr.append(di, alreadyLoaded=True)

        # Load classifier from hdf5
        fileName = str(os.path.join(self.h5_directory.get_absolute_path(), self.classifier_file_name.value))
        
        hf = h5py.File(fileName,'r')
        temp = hf['classifiers'].keys()
        # If hf is not closed this leads to an error in win64 and mac os x
        hf.close()
        del hf
        
        classifiers = []
        for cid in temp:
            cidpath = 'classifiers/' + cid
            try:
                classifiers.append(ClassifierRandomForest.deserialize(fileName, cidpath))
            except:
                classifiers.append(ClassifierRandomForest.loadRFfromFile(fileName, cidpath))

        self.dataMgr.module["Classification"]["classificationMgr"].classifiers = classifiers 
        
        # Restore user selection of feature items from hdf5
        featureItems = []
        f = h5py.File(fileName,'r')
        for fgrp in f['features'].values():
            featureItems.append(FeatureBase.deserialize(fgrp))
        f.close()
        del f

        # Create FeatureMgr
        fm = FeatureMgr(self.dataMgr, featureItems)

        # Compute features
        fm.prepareCompute(self.dataMgr)
        fm.triggerCompute()
        fm.joinCompute(self.dataMgr)
        
        # Predict with loaded classifier
        
        classificationPredict = ClassifierPredictThread(self.dataMgr)
        classificationPredict.start()
        classificationPredict.wait()
        
        # Produce output image and select the probability map
        probMap = classificationPredict._prediction[0][0,0,:,:, int(self.class_sel.value)]
        # probMap = classificationPredict._prediction[0]
        temp_image = cpi.Image(probMap, parent_image=image)
        workspace.image_set.add(self.output_image.value, temp_image)
        workspace.display_data.source_image = image.pixel_data
        workspace.display_data.dest_image = probMap

    def is_interactive(self):
        return False
    
    def display(self, workspace):
        figure = workspace.create_or_find_figure(subplots = (2,1))
        source_image = workspace.display_data.source_image
        dest_image = workspace.display_data.dest_image
        if source_image.ndim == 3:
            src_plot = figure.subplot_imshow_color(
                0, 0, source_image, title = self.image_name.value)
        else:
            src_plot = figure.subplot_imshow_grayscale(
                0, 0, source_image, title = self.image_name.value)
        figure.subplot_imshow_grayscale(
            1, 0, dest_image, title = self.output_image.value)
        