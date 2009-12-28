'''test_relate.py - test the Relate module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import base64
import numpy as np
from scipy.ndimage import distance_transform_edt
from StringIO import StringIO
import unittest
import zlib

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.relate as R

PARENT_OBJECTS = 'parentobjects'
CHILD_OBJECTS = 'childobjects'
MEASUREMENT = 'Measurement'

class TestRelate(unittest.TestCase):
    def make_workspace(self, parents, children, fake_measurement=False):
        '''Make a workspace for testing Relate'''
        pipeline = cpp.Pipeline()
        if fake_measurement:
            class FakeModule(cpm.CPModule):
                def get_measurement_columns(self, pipeline):
                    return [(CHILD_OBJECTS, MEASUREMENT, cpmeas.COLTYPE_FLOAT)]
            module = FakeModule()
            module.module_num = 1
            pipeline.add_module(module)
        module = R.Relate()
        module.parent_name.value = PARENT_OBJECTS
        module.sub_object_name.value = CHILD_OBJECTS
        module.find_parent_child_distances.value = R.D_NONE
        module.module_num = 2 if fake_measurement else 1
        pipeline.add_module(module)
        object_set = cpo.ObjectSet()
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(pipeline,
                                  module,
                                  image_set,
                                  object_set,
                                  cpmeas.Measurements(),
                                  image_set_list)
        o = cpo.Objects()
        o.segmented = parents
        object_set.add_objects(o, PARENT_OBJECTS)
        o = cpo.Objects()
        o.segmented = children
        object_set.add_objects(o, CHILD_OBJECTS)
        return workspace, module
    
    def features_and_columns_match(self, workspace):
        module = workspace.module
        pipeline = workspace.pipeline
        measurements = workspace.measurements
        object_names = [x for x in measurements.get_object_names()
                        if x != cpmeas.IMAGE]
        features = [[feature  
                     for feature in measurements.get_feature_names(object_name)
                     if feature != MEASUREMENT]
                    for object_name in object_names]
        columns = module.get_measurement_columns(pipeline)
        self.assertEqual(sum([len(f) for f in features]), len(columns))
        for column in columns:
            index = object_names.index(column[0])
            self.assertTrue(column[1] in features[index])
        
    def test_01_01_load_matlab_v4(self):
        '''Load a Matlab pipeline with a version 4 relate module'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0'
                'sSU1RyM+zUggpTVXwKs1TMDRTMLCwMjGwMjBWMDIwsFQgGTAwevryMzAwHG'
                'ViYKiY8zbidt5lA4my3UtuLZBKFhXyUHg6qdsw6uBRve2GEhMXrL2WJSRkn'
                'vdKrN9knVB6VqWSXLM3CDHPVl37uNXS+w4zh/u0r9b/9v1/cPh9ijTDh7+C'
                'DKLX8jZFsYUf7XkyKWDnHccpSfkSMw4w8C/+l79mnrP84Xnd5xZKL2I443U'
                'nv5r50ty+9PX3Jte9WKQm09kcmdm2WO5T8zUFNrf1tyua7vD9a3LYdOTwuQ'
                'uqfAYflWqO6H8/90VI6+XdtK/H91bKeAXJ8IRlO9nL13M8e9w7Zcdbjsfli'
                '265Cv9685XF6oqU46uaRzbsXPN8O9tP68rz90eu2Hxjo+rD7Up/VvZPUP8x'
                '+XjfOQnL+Td1CmtiG82FKpjnyGx9q1OlPSlOz9z5uGtE/93VF/TPzefK96r'
                'QM39VrV+5p/QBv+gPa6F5p9WPfOLYekjh1oTKZ/9779ZOZd0+8UBuA3/Oe7'
                'GDLz+FH51cen/lHw31ilMLPMW6o0vuTLO9vzEu9tvqindeNjF5FxW2+MVea'
                'l5XJnf338pHMV4fpjgsVsi7+e306yX7rmu0nJbeLi5Z+Gln5P315l0PSo5H'
                'H51cxndLzjWu4eUXsW8pTztvb+fa91tIfvrG68en75B3un5I2WLOXaHrN8/'
                'vqVTol3VXVH3zqGtu078lH1O+ut9z/z3r7YW3/nL7FvIvb2SOVfmwRit/3a'
                '99C7rVV9tXfj+965fy+f7vMutKbFO3Rn/fa37t99U9P63jz3C7fn76XsXoi'
                '6dFnJ3/rb8Tv+hNcH7u4vp56YH0jXc6j3E/ufx0fZf62/1nbf6Fpy1dmx76'
                'e/vt//H/919d9e25nsexiadVluh9u7CVdxfHvh250//VxJ2+b1l5/no5y7f'
                'ch4fmH53PLsQ//a3dJw05n9t2cp3lh1w/PP2ZX9kW43Hxq01U4JGtZ5bcjT'
                'p75Wu9Qrfd5ijTFPtQC+VX/3/sODrvlXDff8687XXz7hnY6b22+ifOIxZ/F'
                'wAIq5Eo')
        #
        # The Relate module is the fourth in the pipeline:
        # Children = Speckles
        # Parents = Nuclei
        # Calculate per-parent means = yes
        #
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),4)
        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module,R.Relate))
        self.assertEqual(module.sub_object_name.value, "Speckles")
        self.assertEqual(module.parent_name.value, "Nuclei")
        self.assertTrue(module.wants_per_parent_means.value)
    
    def test_01_02_load_v1(self):
        '''Load a pipeline with a version 1 relate module'''
        data = ('eJztm1tP2zAUx90LCIY0sUnTmHixeJomiNINNuBlKTCgEzdBxba3palbvLl'
                '2lTiM7hPsI+1xH2WPe9xHWAwpSb1AQkjbFBIpSo/r3/n7HF+aS7Nbru6U1+'
                'CSosLdcnWhgQmCB0TnDWa2ViHl83DdRDpHdcjoKqzaCL63KSy9hury6uLK6'
                'qIKX6rqCoi35Sq7D53D9BsAxp3jhLPn3a/GXDvn24V9hDjHtGmNgSKYcct/'
                'OfuxbmK9RtCxTmxkeRLd8gptsGqnffnVLqvbBO3pLX9lZ9uzWzVkWvuNLuh'
                '+fYDPEDnC35EUQrfaITrFFmbU5V3/cumlLuOSrsjDj2deHnJSHorOPusrF/'
                'W3gVe/GJC3R776066NaR2f4rqtE4hbevOyFcLfcoi/CcmfsPdN3FxzUi54L'
                'YSflPhJl98yEaK3bMdRGxlfnd6K0o5piRd7FZ3xhXdnusFhS+fGSZR2jEl+'
                'hL2OCLESiuO2+moIn+vhc+BVxHYH6ZbU+UU1IV4L4R9IvLA3GKSMQ9tyJ2i'
                'SfsLyWOjxUwCfnFl1G25Q7c73+MmDPTZaXNj691TKk7A3UEO3CYcVsfjBDW'
                'wigzOzM5R+9nPjEtfdutyke4wzrxUQbT5MSfkS9j63bLhFWE0nl/pJ+elX/'
                'HHHmZy3UkS9KFwa4pPHoTpfSoxLMr5ij14RbJQPKnE4VVFLSXE3iW8Y51EV'
                'yhG1MO/42pGUn0H3832Nf49RFGf+l9R460YQl4b45PV0KaJeFC4N8Y3KeYQ'
                'c32alut7PvEThbhKfFqIX9Xr0vvfnqMZ31+dj3Pic85t4530B3CDi00K4Ub'
                'xuScP1QBQuDfGlcX72s/+UpeS4QcSnhbQzaF5VvzFoEN2y3DvQSfoZdPzbI'
                'e0Oug//AeHmiXiscioeIFAD9dFf2sZD0P3LTWaipslsWk/eT9w4fs54XE7i'
                'gp7HDDJf5w9vRKDtwfsJmoes9gUZ3HM06Pnk04eY1lG7j/6GnZ+MG/7vV9a'
                'fGZdxGZdxyXGaj4v6vw3v/OXiZ3qU4s24jMu40ePCriceg951StjM5gRT9N'
                '8FRT/8ZetvxmXc/eU0H5eG+0mjkreMy7iMy7i7wv3OeZz8/ELY/uftov5nn'
                '07Qev8C9K73wjYQIW2TifeGTKV1/nKLpRCm1y/eLlF2nI8V34smQqcdoqNJ'
                'OtpVOriOKMeNTtt01GzOWjrHhlJxSw+c0nK3dJi6H0N05yTduat0TUR0jpT'
                'D84OXT//4mAzw7+/nvGM9mZ0du25cAdA7nrxx9vdtHL18oZATnP9/sVMhXB'
                'H0jm/B/wE3G8/Pr6nfjTGt9f8Be1L4ug==')
        fd = StringIO(zlib.decompress(base64.b64decode(data)))
        pipeline = cpp.Pipeline()
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()),4)
        module = pipeline.modules()[3]
        self.assertTrue(isinstance(module,R.Relate))
        self.assertEqual(module.sub_object_name.value, "Speckles")
        self.assertEqual(module.parent_name.value, "Cells")
        self.assertFalse(module.wants_per_parent_means.value)
    
    def test_02_01_relate_zeros(self):
        '''Relate a field of empty parents to empty children'''
        labels = np.zeros((10,10),int)
        workspace, module = self.make_workspace(labels, labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS, 
                                               "Parent_%s"%PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 0)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count"%
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 0)
        self.features_and_columns_match(workspace)
    
    def test_02_01_relate_one(self):
        '''Relate one parent to one child'''
        parent_labels = np.ones((10,10),int)
        child_labels = np.zeros((10,10),int)
        child_labels[3:5,4:7] = 1
        workspace, module = self.make_workspace(parent_labels, child_labels)
        module.wants_per_parent_means.value = False
        module.run(workspace)
        m = workspace.measurements
        parents_of = m.get_current_measurement(CHILD_OBJECTS, 
                                               "Parent_%s"%PARENT_OBJECTS)
        self.assertEqual(np.product(parents_of.shape), 1)
        self.assertEqual(parents_of[0],1)
        child_count = m.get_current_measurement(PARENT_OBJECTS,
                                                "Children_%s_Count"%
                                                CHILD_OBJECTS)
        self.assertEqual(np.product(child_count.shape), 1)
        self.assertEqual(child_count[0],1)
        self.features_and_columns_match(workspace)
    
    def test_03_01_mean(self):
        '''Compute the mean for two parents and four children'''
        i,j = np.mgrid[0:20,0:20]
        parent_labels = (i/10 + 1).astype(int) 
        child_labels  = (i/10).astype(int) + (j/10).astype(int) * 2 + 1
        workspace, module = self.make_workspace(parent_labels, child_labels,
                                                fake_measurement=True)
        module.wants_per_parent_means.value = True
        m = workspace.measurements
        m.add_measurement(CHILD_OBJECTS,MEASUREMENT, 
                          np.array([1.0,2.0,3.0,4.0]))
        expected = np.array([2.0, 3.0])
        module.run(workspace)
        name = "Mean_%s_%s"%(CHILD_OBJECTS, MEASUREMENT)
        data = m.get_current_measurement(PARENT_OBJECTS, name)
        self.assertTrue(np.all(data==expected))
        self.features_and_columns_match(workspace)
        
    def test_04_00_distance_empty(self):
        '''Make sure we can handle labels matrices that are all zero'''
        empty_labels = np.zeros((10,20),int)
        some_labels = np.zeros((10,20),int)
        some_labels[2:7,3:8] = 1
        some_labels[3:8,12:17] = 2
        for parent_labels, child_labels, n in ((empty_labels, empty_labels,0),
                                               (some_labels, empty_labels, 0),
                                               (empty_labels, some_labels, 2)):
            workspace,module = self.make_workspace(parent_labels, child_labels)
            self.assertTrue(isinstance(module, R.Relate))
            module.find_parent_child_distances.value = R.D_BOTH
            module.run(workspace)
            self.features_and_columns_match(workspace)
            meas = workspace.measurements
            for feature in (R.FF_CENTROID, R.FF_MINIMUM):
                m = feature % PARENT_OBJECTS
                v = meas.get_current_measurement(CHILD_OBJECTS, m)
                self.assertEqual(len(v), n)
                if n > 0:
                    self.assertTrue(np.all(np.isnan(v)))
        
    def test_04_01_distance_centroids(self):
        '''Check centroid-centroid distance calculation'''
        i,j = np.mgrid[0:14,0:30]
        parent_labels = (i>=7) * 1 + (j>=15) * 2 + 1
        # Centers should be at i=3 and j=7
        parent_centers = np.array([[3,7],[10,7],[3,22],[10,22]], float)
        child_labels = np.zeros(i.shape)
        np.random.seed(0)
        # Take 12 random points and label them
        child_centers = np.random.permutation(np.prod(i.shape))[:12]
        child_centers = np.vstack((i.flatten()[child_centers], 
                                   j.flatten()[child_centers]))
        child_labels[child_centers[0],child_centers[1]] = np.arange(1,13)
        parent_indexes = parent_labels[child_centers[0],
                                       child_centers[1]] -1
        expected = np.sqrt(np.sum((parent_centers[parent_indexes,:] - 
                                   child_centers.transpose())**2,1))

        workspace,module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, R.Relate))
        module.find_parent_child_distances.value = R.D_CENTROID
        module.run(workspace)
        self.features_and_columns_match(workspace)
        meas = workspace.measurements
        v = meas.get_current_measurement(CHILD_OBJECTS,
                                         R.FF_CENTROID % PARENT_OBJECTS)
        self.assertEqual(v.shape[0], 12)
        self.assertTrue(np.all(np.abs(v - expected) < .0001))
        
    def test_04_02_distance_minima(self):
        '''Check centroid-perimeter distance calculation'''
        i,j = np.mgrid[0:14,0:30]
        #
        # Make the objects different sizes to exercise more code
        #
        parent_labels = (i>=6) * 1 + (j>=14) * 2 + 1
        child_labels = np.zeros(i.shape)
        np.random.seed(0)
        # Take 12 random points and label them
        child_centers = np.random.permutation(np.prod(i.shape))[:12]
        child_centers = np.vstack((i.flatten()[child_centers], 
                                   j.flatten()[child_centers]))
        child_labels[child_centers[0],child_centers[1]] = np.arange(1,13)
        #
        # Measure the distance from the child to the edge of its parent.
        # We do this using the distance transform with a background that's
        # the edges of the labels
        #
        background = ((i != 0) & (i != 5) & (i !=6) & (i != 13) &
                      (j != 0) & (j != 13) & (j != 14) & (j != 29))
        d = distance_transform_edt(background)
        expected = d[child_centers[0], child_centers[1]]

        workspace,module = self.make_workspace(parent_labels, child_labels)
        self.assertTrue(isinstance(module, R.Relate))
        module.find_parent_child_distances.value = R.D_MINIMUM
        module.run(workspace)
        self.features_and_columns_match(workspace)
        meas = workspace.measurements
        v = meas.get_current_measurement(CHILD_OBJECTS,
                                         R.FF_MINIMUM % PARENT_OBJECTS)
        self.assertEqual(v.shape[0], 12)
        self.assertTrue(np.all(np.abs(v - expected) < .0001))
