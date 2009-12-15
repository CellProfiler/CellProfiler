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
        self.assertFalse(module.wants_discard_edge)
        self.assertFalse(module.wants_discard_primary)
    
    def test_01_03_load_v3(self):
        data = ('eJztW+Fv2kYUPxKSLauUZdK0TtMq3Yd+aKLENbRR02hroaHZkBqCCmo3pVnr'
                '2Afcau4s+5zCpv0f+xP3sX/CfGCwOTmxMQac1lYs8o77vd+7d++9Oxv7pNx8'
                'UX4G9yUZnpSbey2sI1jXFdaiZvcQErYLj0ykMKRBSg5h00awglRY2IeFwuED'
                '5+8hLMryYxDvyFVPNp2P0s8ArDufXzrnivvVmivnfCeXG4gxTNrWGsiD7932'
                'j875SjGxcqGjV4puI8ujGLVXSYs2+8b4qxOq2TqqKV1/Z+eo2d0LZFqnrRHQ'
                '/bqOe0hv4L+QMIRRt5foEluYEhfv6hdbx7yUCbyNDv1wbDrmCPqfKUztNJgz'
                'A5Pt3G+9Hz2/5QS/5Z3zjq+d9/8VeP3zAX7+xtd/y5Ux0fAl1mxFh7irtMdW'
                'c31yiL7VCX2roFIrD3AHIbh1wQ4u12xVR3jIWwrBbwp4Lh9jnSETadPo2RL0'
                '8LOJemzveU9RGezyqUliPGH4NQHP5SOk6xaI74/nPUMh2qQ/wuYzN6EnBx6A'
                '+PYX5N2HckT7bwl4LtdNaihthTnJNWjnet6F6JEEPdLYD5Be/IlUZsGLPlSg'
                'ZSAVt7BT88ggbyFtQYPnvxXN3q8EHi5XKCSUQdty8zhO/vzuZF8U/g2Bn8tH'
                'fUYNXbG6IDr/VfMdhluZwK2AGp0NFzbeoPg4ZZYNf9HphaJHHu9V/GF187bA'
                'z+UKaim2zmCVF01YwaYTXtTszzTv0+IKkpzYfK0LuNExwm24n0nOV5z1Spbk'
                'wbFbcP/x2RWm7ztBH5cbHdMm7xEZ1slTm+mYIGu2/J3XPOYncHnui8Is8ziv'
                'fJl23qa1oyDPFr/zqm9R5icibj+p8SWZ10HrTpUwRCzM+j49ceKjgVRKNMXs'
                'eyk483ji7Ns0WWK4law/pq0HckA9SDK+r9q3J5GHUeK7Rgma5/jEfU0hJu5R'
                'RFzS62sS/kxyvxm0njc/UKg6+03LzdRZxtsJ4X8k8HP5j3tP6z/xGxnoibSz'
                '/ZZLr51Lpidn5b36+Zm89/j87+I/228t/kUDO70GbduRxvu1wMfluulcHvuq'
                'U9w69xrhdoffbrnkNxaIOrrsn8V/v4XYcVewg8vSztmbN/fPuXsqrhPHDS9t'
                'wuW7QXYlGVdB11HH1ERtk9pEm90vsep/0av/i9r/BeHmvW7HuV5My3inrePF'
                'mOMbrcNh/o16Pypt68E81+U01/9lruefQl1ftB8WtQ+Wpf2l2znv+zBJ7tsW'
                'jUvLfitt8zzvfVTa8/ZTq0vi/mV/SXb++4OHywm4oN8fFxnfgx8reYAb0fUE'
                '1cPhD0OeoiT1LLLO+fghJhoybpC+m5i3acaVQDL5dFPGm+E+T1wJXB/nW2Ay'
                'zvnpxfmwDN2k8Wa4DJfl1c3HlcBy/RvGn+0PMtw8cCWw3LjPcJ8nrgSyuEvb'
                'fdk03Q8ohdifrYcZLsNluAyX4TJcenD/5TxcTsBx2f98C+//zscTtM7v+Ppv'
                'ubKKdN0wKX9v0pS6g5f7LEmnijZ8W0564fxb9b04x3mMEJ6SwFO6igdriDDc'
                '6hv84UOb0a7CsCpV3Vb+SGJ51Mp5OyG8BwLvQRivNXooe8w5fkw7Cl9R4Cte'
                'xYcGL2lR0+qYmLyXhu9snZqNgej51R8nGwF8/vlecaTbd7794rr4AmAyrrx4'
                '+/g0Dl8+v5rbBJPPHd4KweXBZJwP4hpMF9f3ruk/GmOa+0/r55xzzOonjyc/'
                'tmmoP539/we1OWvt')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cpmi2.IdentifySecondary))
        self.assertTrue(module.wants_discard_edge)
        self.assertTrue(module.wants_discard_primary)
        self.assertEqual(module.new_primary_objects_name, "FilteredNuclei")
        
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

    def test_07_01_measurements_no_new_primary(self):
        module = cpmi2.IdentifySecondary()
        for discard_edge in (True, False):
            module.wants_discard_edge.value = discard_edge
            module.wants_discard_primary.value = False
            module.primary_objects.value = "Primary"
            module.objects_name.value = "Secondary"
            module.new_primary_objects_name.value = "NewPrimary"
            
            categories = module.get_categories(None, cpm.IMAGE)
            self.assertEqual(len(categories), 2)
            self.assertTrue(all([any([x == y for x in categories])
                                 for y in ("Count", "Threshold")]))
            categories = module.get_categories(None, "Secondary")
            self.assertEqual(len(categories), 2)
            self.assertTrue(all([any([x == y for x in categories])
                                 for y in ("Location","Parent")]))
            categories = module.get_categories(None, "Primary")
            self.assertEqual(len(categories), 1)
            self.assertEqual(categories[0], "Children")
            
            categories = module.get_categories(None, "NewPrimary")
            self.assertEqual(len(categories), 0)
            
            features = module.get_measurements(None, cpm.IMAGE, "Count")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], "Secondary")
            
            features = module.get_measurements(None, cpm.IMAGE, "Threshold")
            threshold_features = ("OrigThreshold", "FinalThreshold",
                                  "WeightedVariance", "SumOfEntropies")
            self.assertEqual(len(features), 4)
            self.assertTrue(all([any([x == y for x in features])
                                 for y in threshold_features]))
            for threshold_feature in threshold_features:
                objects = module.get_measurement_objects(None, cpm.IMAGE,
                                                         "Threshold",
                                                         threshold_feature)
                self.assertEqual(len(objects), 1)
                self.assertEqual(objects[0], "Secondary")
                
            features = module.get_measurements(None, "Primary","Children")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], "Secondary_Count")
            
            features = module.get_measurements(None,"Secondary","Parent")
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], "Primary")
            
            features = module.get_measurements(None, "Secondary", "Location")
            self.assertEqual(len(features), 2)
            self.assertTrue(all([any([x==y for x in features])
                                 for y in ("Center_X","Center_Y")]))
            
            columns = module.get_measurement_columns(None)
            expected_columns = [ (cpm.IMAGE, "Threshold_%s_Secondary"%f, cpm.COLTYPE_FLOAT)
                                 for f in threshold_features]
            expected_columns += [(cpm.IMAGE, "Count_Secondary", cpm.COLTYPE_INTEGER),
                                 ("Primary", "Children_Secondary_Count", cpm.COLTYPE_INTEGER),
                                 ("Secondary", "Location_Center_X", cpm.COLTYPE_FLOAT),
                                 ("Secondary", "Location_Center_Y", cpm.COLTYPE_FLOAT),
                                 ("Secondary", "Parent_Primary", cpm.COLTYPE_INTEGER)]
            self.assertEqual(len(columns), len(expected_columns))
            for column in expected_columns:
                self.assertTrue(any([all([fa == fb 
                                          for fa,fb 
                                          in zip(column, expected_column)])
                                     for expected_column in expected_columns]))

    def test_07_02_measurements_new_primary(self):
        module = cpmi2.IdentifySecondary()
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.primary_objects.value = "Primary"
        module.objects_name.value = "Secondary"
        module.new_primary_objects_name.value = "NewPrimary"
        
        categories = module.get_categories(None, cpm.IMAGE)
        self.assertEqual(len(categories), 2)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Count", "Threshold")]))
        categories = module.get_categories(None, "Secondary")
        self.assertEqual(len(categories), 2)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Location","Parent")]))
        categories = module.get_categories(None, "Primary")
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], "Children")
        
        categories = module.get_categories(None, "NewPrimary")
        self.assertEqual(len(categories), 3)
        self.assertTrue(all([any([x == y for x in categories])
                             for y in ("Location","Parent","Children")]))
        
        features = module.get_measurements(None, cpm.IMAGE, "Count")
        self.assertEqual(len(features), 2)
        self.assertEqual(features[0], "Secondary","NewPrimary")
        
        features = module.get_measurements(None, cpm.IMAGE, "Threshold")
        threshold_features = ("OrigThreshold", "FinalThreshold",
                              "WeightedVariance", "SumOfEntropies")
        self.assertEqual(len(features), 4)
        self.assertTrue(all([any([x == y for x in features])
                             for y in threshold_features]))
        for threshold_feature in threshold_features:
            objects = module.get_measurement_objects(None, cpm.IMAGE,
                                                     "Threshold",
                                                     threshold_feature)
            self.assertEqual(len(objects), 1)
            self.assertEqual(objects[0], "Secondary")
            
        features = module.get_measurements(None, "Primary","Children")
        self.assertEqual(len(features), 2)
        self.assertTrue(all([any([x==y for x in features])
                            for y in ("Secondary_Count","NewPrimary_Count")]))
        
        features = module.get_measurements(None,"Secondary","Parent")
        self.assertEqual(len(features), 2)
        self.assertTrue(all([any([x==y for x in features])
                             for y in ("Primary","NewPrimary")]))
        
        for oname in ("Secondary","NewPrimary"):
            features = module.get_measurements(None, oname, "Location")
            self.assertEqual(len(features), 2)
            self.assertTrue(all([any([x==y for x in features])
                                 for y in ("Center_X","Center_Y")]))
        
        columns = module.get_measurement_columns(None)
        expected_columns = [ (cpm.IMAGE, "Threshold_%s_Secondary"%f, cpm.COLTYPE_FLOAT)
                             for f in threshold_features]
        for oname in ("NewPrimary","Secondary"):
            expected_columns += [(cpm.IMAGE, "Count_%s"%oname, cpm.COLTYPE_INTEGER),
                                 ("Primary", "Children_%s_Count"%oname, cpm.COLTYPE_INTEGER),
                                 (oname, "Location_Center_X", cpm.COLTYPE_FLOAT),
                                 (oname, "Location_Center_Y", cpm.COLTYPE_FLOAT),
                                 (oname, "Parent_Primary", cpm.COLTYPE_INTEGER)]
        expected_columns += [("NewPrimary","Children_Secondary_Count", cpm.COLTYPE_INTEGER),
                             ("Secondary","Parent_NewPrimary", cpm.COLTYPE_INTEGER)]
        self.assertEqual(len(columns), len(expected_columns))
        for column in expected_columns:
            self.assertTrue(any([all([fa == fb 
                                      for fa,fb 
                                      in zip(column, expected_column)])
                                 for expected_column in expected_columns]))
        
    def test_08_01_filter_edge(self):
        labels = np.array([[0,0,0,0,0],
                           [0,0,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,0,0],
                           [0,0,0,0,0]])
        image = np.array([[0, 0,.5, 0,0],
                          [0,.5,.5,.5,0],
                          [0,.5,.5,.5,0],
                          [0,.5,.5,.5,0],
                          [0, 0, 0, 0,0]])
        expected_unedited = np.array([[0,0,1,0,0],
                                      [0,1,1,1,0],
                                      [0,1,1,1,0],
                                      [0,1,1,1,0],
                                      [0,0,0,0,0]])
        
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(image)
        objects = cpo.Objects()
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
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.new_primary_objects_name.value = "newprimary"
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        object_out = workspace.object_set.get_objects("my_objects")
        self.assertTrue(np.all(object_out.segmented == 0))
        self.assertTrue(np.all(object_out.unedited_segmented == expected_unedited))
        
        object_out = workspace.object_set.get_objects("newprimary")
        self.assertTrue(np.all(object_out.segmented == 0))
        self.assertTrue(np.all(object_out.unedited_segmented == labels))

    def test_08_02_filter_unedited(self):
        labels = np.array([[0,0,0,0,0],
                           [0,0,0,0,0],
                           [0,0,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,0,0]])
        labels_unedited = np.array([[0,0,1,0,0],
                                    [0,0,0,0,0],
                                    [0,0,0,0,0],
                                    [0,0,2,0,0],
                                    [0,0,0,0,0]])
        image = np.array([[0, 0,.5, 0,0],
                          [0,.5,.5,.5,0],
                          [0,.5,.5,.5,0],
                          [0,.5,.5,.5,0],
                          [0, 0, 0, 0,0]])
        expected = np.array([[0,0,0,0,0],
                             [0,0,0,0,0],
                             [0,1,1,1,0],
                             [0,1,1,1,0],
                             [0,0,0,0,0]])
        expected_unedited = np.array([[0,0,1,0,0],
                                      [0,1,1,1,0],
                                      [0,2,2,2,0],
                                      [0,2,2,2,0],
                                      [0,0,0,0,0]])
        
        p = cpp.Pipeline()
        o_s = cpo.ObjectSet()
        i_l = cpi.ImageSetList()
        image = cpi.Image(image)
        objects = cpo.Objects()
        objects.unedited_segmented = labels 
        objects.small_removed_segmented = labels
        objects.unedited_segmented = labels_unedited
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
        module.wants_discard_edge.value = True
        module.wants_discard_primary.value = True
        module.new_primary_objects_name.value = "newprimary"
        workspace = cpw.Workspace(p,module,i_s,o_s,m,i_l)
        module.run(workspace)
        object_out = workspace.object_set.get_objects("my_objects")
        self.assertTrue(np.all(object_out.segmented == expected))
        self.assertTrue(np.all(object_out.unedited_segmented == expected_unedited))
        
        object_out = workspace.object_set.get_objects("newprimary")
        self.assertTrue(np.all(object_out.segmented == labels))
        self.assertTrue(np.all(object_out.unedited_segmented == labels_unedited))
        