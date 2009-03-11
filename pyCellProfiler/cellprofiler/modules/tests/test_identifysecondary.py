"""test_identifysecondary - test the IdentifySecondary module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revision$"

import base64
import numpy as np
import unittest
import StringIO

import cellprofiler.modules.identifysecondary as cpmi2
import cellprofiler.modules.identify as cpmi
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpm

class TestIdentifySecondary(unittest.TestCase):
    def test_01_01_load_matlab(self):
        u64data = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBUaHUgRmViIDE5IDE1OjQyOjQyIDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAyAIAAHic7VjLjtMwFHWfGqAalRmBYFZZshiGRIDEkqegErQVHY3E0k3cYOTEVeIMLV/DZ7BgMWu+iQVOm7SuJ2ncJM1ssGRF1/U5vvf4+sbpIQBAvw9Amz8PeK+DZWtFdk3ooT1CjGHX9lugCR5E41e8X0APwzFBF5AEyAerFo/33Ak9n09XP32iVkBQHzriZN76gTNGnj+YxMDo5yGeITLCPxDYbPG0z+gS+5i6ET7il0dX61ImrXvI+8/OWoeapEOD92NhPJz/EqznNxN06wrzu1E/RzP2+N0MmkxzIDO/hjwvMngOJJ7QHnjYfs2lVsG3JXx7obNJEF7GoWfg6xv4OvhuLHFF183CtyR8aL9BhPhADV9Ut6T1Df30mQ7U9v+OhA/toUen0IaMJ+ViXIXntsQT2m+p5lKmBX50IFT2sbHB0wBf+BkoQ8eq/JfzsE/z78OA+YH2ntAxJKv1q4ojbR+ycLUNXA0YJftdlo67xq+fGpXqn6ZjFq65gWsC/Uw3bjJvYt12PTeGnk+npwV1yoF7XmV+35J4QrvnMuT6mM2BOo+qPx8yeO5KPKGNXQtfYiuARMMOtFe3mDLjy1uH91W/k/x+FTDKL1DYFPxW9V/O67MC+m3zIw+f7cG5b0KCBJ6875Oi+Cz/VfNc1Y+8+VG2H7Ie7hNYyj4U1SHGX7W2f6eIdWOXfEzScVFkbI8GU3WepHsCHX9DJlsTiX7lqYMCn8ZrIprukS8r3qTvvLVuS7q0/W9LfHGL+eoCTn5m+VXWft5UfHl5/sdTTTxpz6zzdyT5G9o0YAS76NoB3GXdfT33FU+3ll7H5fdJ3vP+kUKrJ1wUVeK5J/GEds9CLsOT+dDDjnjnyVtvY74RMqlrQS/t/innp6xLnVtHx53SvvtU1mt1Gtf+N8nCNaM5f09+n/x5uFz3F9ht/x9tmR+3qub/A5FILLM='
        data = base64.b64decode(u64data)
        p = cpp.Pipeline()
        fd = StringIO.StringIO(data)
        p.load(fd)
        self.assertTrue(len(p.modules())==3)
        module = p.modules()[2]
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
        self.assertEqual(module.primary_objects.value,"Nuclei")
        self.assertEqual(module.objects_name.value,"Cells")
        self.assertEqual(module.method.value,cpmi2.M_PROPAGATION)
        self.assertEqual(module.image_name.value,"OrigBlue")
        self.assertEqual(module.threshold_method.value,"Otsu Global")
        self.assertEqual(module.threshold_correction_factor.value, 1)
        self.assertEqual(module.threshold_range.min, 0)
        self.assertEqual(module.threshold_range.max, 1)
        self.assertEqual(module.distance_to_dilate.value, 10)
        self.assertEqual(module.regularization_factor.value, 0.05)
    
    def test_02_01_zeros_propagation(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(np.zeros((10,10)))
        objects = cpo.Objects()
        objects.unedited_segmented = np.zeros((10,10),int)
        objects.small_removed_segmented = np.zeros((10,10),int)
        objects.segmented = np.zeros((10,10),int)
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method = cpmi2.M_PROPAGATION
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],0)
    
    def test_02_02_one_object_propagation(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,10))
        img[2:7,2:7] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method = cpmi2.M_PROPAGATION
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],1)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,10),int)
        expected[2:7,2:7] = 1
        self.assertTrue(np.all(objects_out.segmented==expected))

    def test_02_03_two_objects_propagation_image(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,20))
        img[2:7,2:7] = .3
        img[2:7,7:17] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,20),int)
        labels[3:6,3:6] = 1
        labels[3:6,13:16] = 2
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method.value = cpmi2.M_PROPAGATION
        module.regularization_factor.value = 0 # propagate by image
        module.threshold_method.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .2
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],2)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,10),int)
        expected[2:7,2:7] = 1
        expected[2:7,7:17] = 2
        mask = np.ones((10,10),bool)
        mask[:,7:9] = False
        self.assertTrue(np.all(objects_out.segmented[mask]==expected[mask]))

    def test_02_04_two_objects_propagation_distance(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,20))
        img[2:7,2:7] = .3
        img[2:7,7:17] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,20),int)
        labels[3:6,3:6] = 1
        labels[3:6,13:16] = 2
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method.value = cpmi2.M_PROPAGATION
        module.regularization_factor.value = 1000 # propagate by distance
        module.threshold_method.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .2
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],2)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,20),int)
        expected[2:7,2:10] = 1
        expected[2:7,10:17] = 2
        mask = np.ones((10,20),bool)
        mask[:,9:11] = False
        self.assertTrue(np.all(objects_out.segmented[mask]==expected[mask]))
    
    def test_03_01_zeros_watershed_gradient(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(np.zeros((10,10)))
        objects = cpo.Objects()
        objects.unedited_segmented = np.zeros((10,10),int)
        objects.small_removed_segmented = np.zeros((10,10),int)
        objects.segmented = np.zeros((10,10),int)
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method = cpmi2.M_WATERSHED_G
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],0)
    
    def test_03_02_one_object_watershed_gradiant(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,10))
        img[2:7,2:7] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method = cpmi2.M_WATERSHED_G
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],1)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,10),int)
        expected[2:7,2:7] = 1
        self.assertTrue(np.all(objects_out.segmented==expected))
        self.assertTrue("Location_Center_X" in m.get_feature_names("my_objects"))
        values = m.get_current_measurement("my_objects","Location_Center_X")
        self.assertEqual(np.product(values.shape),1)
        self.assertEqual(values[0],4)
        self.assertTrue("Location_Center_Y" in m.get_feature_names("my_objects"))
        values = m.get_current_measurement("my_objects","Location_Center_Y")
        self.assertEqual(np.product(values.shape),1)
        self.assertEqual(values[0],4)

    def test_03_03_two_objects_watershed_gradient(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,20))
        # There should be a gradient at :,7 which should act
        # as the watershed barrier
        img[2:7,2:7] = .3
        img[2:7,7:17] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,20),int)
        labels[3:6,3:6] = 1
        labels[3:6,13:16] = 2
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method.value = cpmi2.M_WATERSHED_G
        module.threshold_method.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .2
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],2)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,20),int)
        expected[2:7,2:7] = 1
        expected[2:7,7:17] = 2
        mask = np.ones((10,20),bool)
        mask[:,7:9] = False
        self.assertTrue(np.all(objects_out.segmented[mask]==expected[mask]))
    
    def test_04_01_zeros_watershed_image(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(np.zeros((10,10)))
        objects = cpo.Objects()
        objects.unedited_segmented = np.zeros((10,10),int)
        objects.small_removed_segmented = np.zeros((10,10),int)
        objects.segmented = np.zeros((10,10),int)
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method = cpmi2.M_WATERSHED_I
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],0)
    
    def test_04_02_one_object_watershed_image(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,10))
        img[2:7,2:7] = .5
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method = cpmi2.M_WATERSHED_I
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],1)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,10),int)
        expected[2:7,2:7] = 1
        self.assertTrue(np.all(objects_out.segmented==expected))

    def test_04_03_two_objects_watershed_image(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,20))
        # There should be a saddle at 7 which should serve
        # as the watershed barrier
        x,y = np.mgrid[0:10,0:20]
        img[2:7,2:7] = .05 * (7-y[2:7,2:7])
        img[2:7,7:17] = .05 * (y[2:7,7:17]-6)
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,20),int)
        labels[3:6,3:6] = 1
        labels[3:6,13:16] = 2
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method.value = cpmi2.M_WATERSHED_I
        module.threshold_method.value = cpmi.TM_MANUAL
        module.manual_threshold.value = .01
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],2)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,20),int)
        expected[2:7,2:7] = 1
        expected[2:7,7:17] = 2
        mask = np.ones((10,20),bool)
        mask[:,7] = False
        self.assertTrue(np.all(objects_out.segmented[mask]==expected[mask]))
    
    def test_05_01_zeros_distance_n(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(np.zeros((10,10)))
        objects = cpo.Objects()
        objects.unedited_segmented = np.zeros((10,10),int)
        objects.small_removed_segmented = np.zeros((10,10),int)
        objects.segmented = np.zeros((10,10),int)
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method = cpmi2.M_DISTANCE_N
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],0)
    
    def test_05_02_one_object_distance_n(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,10))
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,10),int)
        labels[3:6,3:6] = 1
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method = cpmi2.M_DISTANCE_N
        module.distance_to_dilate.value = 1
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],1)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,10),int)
        expected[2:7,2:7] = 1
        for x in (2,6):
            for y in (2,6):
                expected[x,y] = 0
        self.assertTrue(np.all(objects_out.segmented==expected))

    def test_05_03_two_objects_distance_n(self):
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        img = np.zeros((10,20))
        image = cpi.Image(img)
        objects = cpo.Objects()
        labels = np.zeros((10,20),int)
        labels[3:6,3:6] = 1
        labels[3:6,13:16] = 2
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.segmented = labels
        o_s.add_objects(objects, "primary")
        i_s = i_l.get_image_set(0)
        i_s.add("my_image",image)
        m = cpm.Measurements()
        module = cpmi2.IdentifySecondary()
        module.primary_objects.value="primary"
        module.objects_name.value="my_objects"
        module.image_name.value = "my_image"
        module.method.value = cpmi2.M_DISTANCE_N
        module.distance_to_dilate.value = 100
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],2)
        objects_out = o_s.get_objects("my_objects")
        expected = np.zeros((10,20),int)
        expected[:,:10] = 1
        expected[:,10:] = 2
        self.assertTrue(np.all(objects_out.segmented==expected))
    
