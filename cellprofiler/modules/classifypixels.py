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
    from ilastik.core import dataMgr, featureMgr, classificationMgr
    from ilastik.core.features.featureBase import FeatureBase
    from ilastik.core.classifiers.classifierRandomForestNew import ClassifierRandomForestNew
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
        
        # TODO: workarround, need to get the real scaling factor to 
        # recover raw image domain
        image_ = image.pixel_data * 255
        
        print "Input Image shape", image_.shape
        print "Input Image min", image_.min()
        print "Input Image max", image_.max()
        
        # Create ilastik dataMgr
        self.dataMgr = dataMgr.DataMgr()
        
        # Transform input image to ilastik convention s
        # 3D = (time,x,y,z,channel) 
        # 2D = (time,1,x,y,channel)
        # Note, this work for 2D images right now. Is there a need for 3D
        image_.shape = (1,1) + image_.shape
        
        # Check if image_ has channels, if not add singelton dimension
        if len(image_.shape) == 4:
            image_.shape = image_.shape + (1,)
        
        # Add data item di to dataMgr
        di = dataMgr.DataItemImage.initFromArray(image_, '')
        self.dataMgr.append(di, alreadyLoaded=True)

        # Load classifier from hdf5
        fileName = str(os.path.join(self.h5_directory.get_absolute_path(), self.classifier_file_name.value))
        
        hf = h5py.File(fileName,'r')
        classifiers = []
        for cid in hf['classifiers']:
            classifiers.append(ClassifierRandomForestNew.deserialize(fileName, 'classifiers/' + cid))   
        self.dataMgr.classifiers = classifiers
        
        # Restore user selection of feature items from hdf5
        featureItems = []
        f = h5py.File(fileName,'r')
        for fgrp in f['features'].values():
            featureItems.append(FeatureBase.deserialize(fgrp))

        # Create FeatureMgr
        fm = featureMgr.FeatureMgr(self.dataMgr, featureItems)

        # Compute features
        fm.prepareCompute(self.dataMgr)
        fm.triggerCompute()
        fm.joinCompute(self.dataMgr)
        
        # Predict with loaded classifier
        classificationPredict = classificationMgr.ClassifierPredictThread(self.dataMgr)
        classificationPredict.start()
        classificationPredict.wait()
        
        # Produce output image and select the probability map
        probMap = self.dataMgr[0].prediction[0,0,:,:, int(self.class_sel.value)]
        temp_image = cpi.Image(probMap, parent_image=image)
        workspace.image_set.add(self.output_image.value, temp_image)   
