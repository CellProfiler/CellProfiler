"""test_identifytertiarysubregion.py - test the IdentifyTertiarySubregion module

"""
__version__="$Revision$"

import numpy as np
import unittest

import cellprofiler.modules.identifytertiarysubregion as cpmit
import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpm

PRIMARY = "primary"
SECONDARY = "secondary"
TERTIARY = "tertiary"

class TestIdentifyTertiarySubregion(unittest.TestCase):
    def make_workspace(self,primary_labels,secondary_labels):
        """Make a workspace that has objects for the input labels
        
        returns a workspace with the following
            object_set - has object with name "primary" containing
                         the primary labels
                         has object with name "secondary" containing
                         the secondary labels
        """
        isl = cpi.ImageSetList()
        module = cpmit.IdentifyTertiarySubregion()
        module.primary_objects_name.value = PRIMARY
        module.secondary_objects_name.value = SECONDARY
        module.subregion_objects_name.value = TERTIARY
        workspace = cpw.Workspace(cpp.Pipeline(),
                                  module,
                                  isl.get_image_set(0),
                                  cpo.ObjectSet(),
                                  cpm.Measurements(),
                                  isl)
        
        for labels, name in ((primary_labels,PRIMARY),
                             (secondary_labels, SECONDARY)):
            objects = cpo.Objects()
            objects.segmented = labels
            workspace.object_set.add_objects(objects, name)
        return workspace
    
    def test_00_00_zeros(self):
        """Test IdentifyTertiarySubregion on an empty image"""
        primary_labels = np.zeros((10,10),int)
        secondary_labels = np.zeros((10,10),int)
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue("Image" in measurements.get_object_names())
        count_feature = "Count_%s"%(TERTIARY)
        self.assertTrue(count_feature in 
                        measurements.get_feature_names("Image"))
        value = measurements.get_current_measurement("Image",count_feature)
        self.assertEqual(np.product(value.shape),1)
        self.assertEqual(value[0], 0)
        self.assertTrue(TERTIARY in workspace.object_set.get_object_names())
        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(np.all(output_objects.segmented == primary_labels))
    
    def test_01_01_one_object(self):
        """Test creation of a single tertiary object"""
        primary_labels = np.zeros((10,10),int)
        secondary_labels = np.zeros((10,10),int)
        primary_labels[3:6,4:7] = 1
        secondary_labels[2:7,3:8] = 1
        expected_labels = np.zeros((10,10),int)
        expected_labels[2:7,3:8] = 1
        expected_labels[4,5] = 0
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        self.assertTrue("Image" in measurements.get_object_names())
        count_feature = "Count_%s"%(TERTIARY)
        self.assertTrue(count_feature in 
                        measurements.get_feature_names("Image"))
        value = measurements.get_current_measurement("Image",count_feature)
        self.assertEqual(np.product(value.shape),1)
        self.assertEqual(value[0], 1)
        
        self.assertTrue(TERTIARY in measurements.get_object_names())
        child_count_feature = "Children_%s_Count"%(TERTIARY)
        for parent_name in (PRIMARY,SECONDARY):
            parents_of_feature = ("Parent_%s"%(parent_name))
            self.assertTrue(parents_of_feature in
                            measurements.get_feature_names(TERTIARY))
            value = measurements.get_current_measurement(TERTIARY, 
                                                         parents_of_feature)
            self.assertTrue(np.product(value.shape), 1)
            self.assertTrue(value[0], 1)
            self.assertTrue(child_count_feature in
                            measurements.get_feature_names(parent_name))
            value = measurements.get_current_measurement(parent_name,
                                                         child_count_feature)
            self.assertTrue(np.product(value.shape),1)
            self.assertTrue(value[0], 1)
        
        for axis, expected in (("X",4),("Y",5)):
            feature = "Location_Center_%s"%(axis)
            self.assertTrue(feature in measurements.get_feature_names(TERTIARY))
            value = measurements.get_current_measurement(TERTIARY, feature)
            self.assertTrue(np.product(value.shape),1)
            self.assertEqual(value[0],expected)

        self.assertTrue(TERTIARY in workspace.object_set.get_object_names())
        output_objects = workspace.object_set.get_objects(TERTIARY)
        self.assertTrue(np.all(output_objects.segmented == expected_labels))
    
    def test_01_02_two_objects(self):
        """Test creation of two tertiary objects"""
        primary_labels = np.zeros((10,20),int)
        secondary_labels = np.zeros((10,20),int)
        expected_primary_parents = np.zeros((10,20),int)
        expected_secondary_parents = np.zeros((10,20),int)
        centers = ((4,5,1,2),(4,15,2,1))
        for x,y,primary_label,secondary_label in centers:
            primary_labels[x-1:x+2,y-1:y+2] = primary_label
            secondary_labels[x-2:x+3,y-2:y+3] = secondary_label
            expected_primary_parents[x-2:x+3,y-2:y+3] = primary_label
            expected_primary_parents[x,y] = 0
            expected_secondary_parents[x-2:x+3,y-2:y+3] = secondary_label
            expected_secondary_parents[x,y] = 0
         
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        count_feature = "Count_%s"%(TERTIARY)
        value = measurements.get_current_measurement("Image",count_feature)
        self.assertEqual(value[0], 2)
        
        child_count_feature = "Children_%s_Count"%(TERTIARY)
        output_labels = workspace.object_set.get_objects(TERTIARY).segmented
        for parent_name,idx,parent_labels in ((PRIMARY,2, expected_primary_parents),
                                              (SECONDARY,3,expected_secondary_parents)):
            parents_of_feature = ("Parent_%s"%(parent_name))
            cvalue = measurements.get_current_measurement(parent_name,
                                                          child_count_feature)
            self.assertTrue(np.all(cvalue==1))
            pvalue = measurements.get_current_measurement(TERTIARY, 
                                                          parents_of_feature)
            for value in (pvalue,cvalue):
                self.assertTrue(np.product(value.shape), 2)
            #
            # Make an array that maps the parent label index to the
            # corresponding child label index
            #
            label_map = np.zeros((len(centers)+1,),int)
            for center in centers:
                label = center[idx]
                label_map[label] = pvalue[center[idx]-1] 
            expected_labels = label_map[parent_labels]
            self.assertTrue(np.all(expected_labels == output_labels))
    
    def test_01_03_overlapping_secondary(self):
        """Make sure that an overlapping tertiary is assigned to the larger parent"""
        expected_primary_parents = np.zeros((10,20),int)
        expected_secondary_parents = np.zeros((10,20),int)
        primary_labels = np.zeros((10,20),int)
        secondary_labels = np.zeros((10,20),int)
        primary_labels[3:6,3:10] = 2
        primary_labels[3:6,10:17] = 1
        secondary_labels[2:7,2:12] = 1
        expected_primary_parents[2:7,2:12]=2
        expected_primary_parents[4,4:12]=0 # the middle of the primary
        expected_primary_parents[4,9]=2    # the outline of primary # 2
        expected_primary_parents[4,10]=2   # the outline of primary # 1
        expected_secondary_parents[expected_primary_parents>0]=1
        workspace = self.make_workspace(primary_labels, secondary_labels)
        module = workspace.module
        module.run(workspace)
        measurements = workspace.measurements
        output_labels = workspace.object_set.get_objects(TERTIARY).segmented
        for parent_name, parent_labels in ((PRIMARY, expected_primary_parents),
                                           (SECONDARY, expected_secondary_parents)):
            parents_of_feature = ("Parent_%s"%(parent_name))
            pvalue = measurements.get_current_measurement(TERTIARY, 
                                                          parents_of_feature)
            label_map = np.zeros((np.product(pvalue.shape)+1,),int)
            label_map[1:]=pvalue.flatten()
            mapped_labels = label_map[output_labels]
            self.assertTrue(np.all(parent_labels == mapped_labels))
            
