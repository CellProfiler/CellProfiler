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
import zlib

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
    
    def test_01_02_load_v2(self):
        data = ('eJztWt1u2zYUlh0nSFZ0a9OLDugNL5suNiQ3xtJgSO3G/fEWu0bjtS'
                'iKbmMk2uZAk4ZEpXGLAnuUPUYv+zi97CNUdCRLZpVIkeM/QAQI+Rzx'
                'O38kzxFl1Sutw8ojUCqooF5p5duYINAkkLeZ2dsDlG+DAxNBjgzA6B'
                '6oMwp+tylQd4FW3Ctpe6UdUFTVB0qylqnVf3Qun+8pyppzXXd61r21'
                '6tKZQBf0EeIc0461quSUn13+Z6e/hCaGxwS9hMRGlq/C49dom7UG/d'
                'GtOjNsghqwFxzstIbdO0am9bztAd3bTXyKyBF+jyQXvGEv0Am2MKMu'
                '3pUvc0d6GZf0ijh8ueXHISPFYcXpdwJ8Mf6Z4o/PhcTtZmD8DZfG1M'
                'An2LAhAbgHOyMrhDw1Qt7KmLwVpdqoDHG7Ebg1yY61YZx1gvCZ3nIE'
                '/oaEF72FTnn+8SnUOehBrnevwo4o/KqEF/QBIsSKGb/MGD6j3FeS69'
                'XU7R1ViRe/axJe0E2T9WEHcmdxDvlJ7agdHv5Zj+m/vH5eO6tvknUX'
                'hcuO4bJKg8Wz8zxckjg/55YNnhJ2DMkozlH79rYkR9BV1IY24aAmNi'
                '2oYhPpnJmDmcZdK6jf4dYknNc83IZ7nWX8wvKeWlCHbVtzfwTsmmf8'
                'wnC5MVxO2K5NYues4yXvH02Nl1/WlXE7BH3QhZQiUpxm3EJwpUnyUj'
                'kCtyH5Kega5YhamA8CcY6S84MkR9BVBijjwLaQL+eydUmLqT+uH5eN'
                'o5pwnzQYRZOsz6T6/ovA/SHFSdB/3X3Y/E08aKP9wi9bfwvqlfMo8Y'
                'K9239TyTffbnmcA0bsHt1/o+YfvP2gbRc/ng0+wg5yyNyKHWd5nn+N'
                'iTuvHnYjcLuS34IWtr9G0HQd2vm4lRcs52DBuy6v6PKqcOBzJsl/5Q'
                'hcWN1pvWNAJ9Cy3Cfkada9uPs4SR5/hXCnK45vJ+KgQnXv/DKJP9Oa'
                'h7A4PGEm6pjMpsb87L4Kfamd4Xl9EeyMU0cWwc4455lFsHN512dpLn'
                'aWI+y8roznRUG3uiZCgQo1D7sXsR5Nuy5fZT36f/Ny7/tm6efw5aBw'
                'tB9fTthzFDv+F+ncFzTr9RbQDzA1UH+K8uaZ3xYZV1auZv0ti78pLs'
                'WluMlx5QAu7v9Cft44S8/TrDebkn5BM5sTTNF3BWKZ4r5o85zWh+XC'
                'pfsmxaW4NF+muBSX4pYfd5rxcfJ7Kvk9qhj/T0BPWH66p4znJ0HriJ'
                'C+ycT3h2ahN/xIzioQBo2zr9QKh87PWuCDNaGnH6GnLOkpn6cHG4hy'
                '3B70TUebzVkPcqwXai636XArHjfp/54X6rWQzqgBzcFI55HHkedtI0'
                'RfMP5Zh/rpzs0L51ueZ3/+vz5Moi+7kh3qC37fcS0ClwvYJJrAf1Iu'
                't87uXjDe83FW478B0PjACw==')
        p = cpp.Pipeline()
        fd = StringIO.StringIO(zlib.decompress(base64.b64decode(data)))
        p.load(fd)
        self.assertTrue(len(p.modules())==3)
        module = p.modules()[2]
        self.assertEqual(module.threshold_method.value, "Otsu Global")
        self.assertEqual(module.two_class_otsu.value, cpmi.O_TWO_CLASS)
        self.assertEqual(module.use_weighted_variance.value,
                         cpmi.O_WEIGHTED_VARIANCE)
        
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
        module.use_outlines.value = False
        module.outlines_name.value = "my_outlines"
        module.method = cpmi2.M_PROPAGATION
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        self.assertTrue("my_objects" in m.get_object_names())
        self.assertTrue("Image" in m.get_object_names())
        self.assertTrue("Count_my_objects" in m.get_feature_names("Image"))
        counts = m.get_current_measurement("Image", "Count_my_objects")
        self.assertEqual(np.product(counts.shape), 1)
        self.assertEqual(counts[0],0)
        columns = module.get_measurement_columns(workspace.pipeline)
        for object_name in (cpm.IMAGE, "my_objects","primary"):
            ocolumns =[x for x in columns if x[0] == object_name]
            features = m.get_feature_names(object_name)
            self.assertEqual(len(ocolumns), len(features))
            self.assertTrue(all([column[1] in features for column in ocolumns]))
        self.assertTrue("my_outlines" not in workspace.get_outline_names())
    
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
        child_counts = m.get_current_measurement("primary","Children_my_objects_Count")
        self.assertEqual(len(child_counts),1)
        self.assertEqual(child_counts[0],1)
        parents = m.get_current_measurement("my_objects","Parent_primary")
        self.assertEqual(len(parents),1)
        self.assertEqual(parents[0],1)

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
    
    def test_06_01_save_outlines(self):
        '''Test the "save_outlines" feature'''
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
        module.use_outlines.value = True
        module.outlines_name.value = "my_outlines"
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
        outlines_out = workspace.image_set.get_image("my_outlines",
                                                     must_be_binary=True).pixel_data
        expected = np.zeros((10,10),int)
        expected[2:7,2:7] = 1
        outlines = expected == 1
        outlines[3:6,3:6] = False
        self.assertTrue(np.all(objects_out.segmented==expected))
        self.assertTrue(np.all(outlines == outlines_out))
