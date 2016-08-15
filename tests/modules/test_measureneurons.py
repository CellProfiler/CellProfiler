'''test_measureneurons.py - test the MeasureNeurons module
'''

import base64
import os
import tempfile
import traceback
import unittest
import zlib
from StringIO import StringIO

import numpy as np
import scipy.ndimage

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw
import cellprofiler.modules.measureneurons as M

IMAGE_NAME = "MyImage"
INTENSITY_IMAGE_NAME = "MyIntensityImage"
OBJECT_NAME = "MyObject"
EDGE_FILE = "my_edges.csv"
VERTEX_FILE = "my_vertices.csv"


class TestMeasureNeurons(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if hasattr(self, "temp_dir"):
            for file_name in (EDGE_FILE, VERTEX_FILE):
                p = os.path.join(self.temp_dir, file_name)
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except:
                    print "Failed to remove %s" % p
                    traceback.print_exc()
            os.rmdir(self.temp_dir)

    def test_01_01_load_matlab(self):
        '''Load a Matlab version of MeasureNeurons'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggH0l6JeQoGZgqGplbGFlZGlgpGBoYGCiQDBkZPX34GBoYyJgaG'
                'ijlPw2PzLxuIlP4y7G1ctYmrc/vTie2GUY0X5MJXcihtXLPossq9AuHtEqpl'
                'U28Enza273L/x/S/Y4Za7+My30dlb1mmrXlh9X1edU1OvDovw4FK7obmF5US'
                'X7lmbK/KkQ4S/8Zg2HbSYk2zVIVc//+uX7w8BjJVL9iqZFoTLk4q7O558TVn'
                'U0WpZ+w7Relfm4qv5LzjDxSI65104EepvP9E8wP3BU9sO/JWeNdRxWUqff0f'
                'r150jTs1ZU/vinpLr1IDEdlVgfmejw/9Z1ryb6voLlPV/2l31l+csrfGvJXP'
                'uPbEnPVMy+c2B599EOJ4/l2TUuFWnxhenhX8pY+q7u4rnf+z5Hl5+J1Kv5jY'
                'SPHE8lvL8132udnEpdT2i8gkfl/hqWFc2RE0Z9PnHzJvz9Z+WJFSoHVg2f9t'
                'cpUW/aIyifEbJ8x+b2vVco8p/fovCUkZ5pJvR5xjl3/LVD+8+I1NnHhP8Guf'
                '32vu3jz+r2LJA47bSjwTBTbPPKJeWXfiYPWqB5zSwuL3k6Y7nz9iJ6Oft/b3'
                'g+XR8iUVJZbdeextsrxyr/ZZ3nboL7Up+XCMcU4r35cJLZsz917uPf84/ouM'
                'yNOFu+23nXW+2rynKTqy3bfU49Hsr1/vzyySvXD1ff5GCfYp/mX9m1/dW3I4'
                '/0Si5gWzwpJXq1qS3UzFfL8t+L7qf7NIul/21X+3p+WlFv4X38evet+59tnc'
                '7ckTrjremGLML7yl5EHKHefH73atuJZ+t/jXRKfrk+yWdJ15PrNJOee0a8vD'
                'i2vkE+L/tWkcWr3wAedvoYMT2E+42SxQidmXzlf0ZYXLtcLVmy7+d75Z81X9'
                '2KX/ZXExut8qqz89frLd8vxv4SnR38Wz/0zepv3f3ukznzkA9yxhQQ==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, M.MeasureNeurons))
        self.assertEqual(module.seed_objects_name, "Soma")
        self.assertEqual(module.image_name, "DNA")

    def test_01_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:8977

MeasureNeurons:[module_num:1|svn_version:\'8401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Seed objects name\x3A:Nucs
    Skeletonized image name\x3A:DNA
    Do you want to save the branchpoint image?:Yes
    Branchpoint image name\x3A:BPImg
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, M.MeasureNeurons))
        self.assertEqual(module.image_name, "DNA")
        self.assertEqual(module.seed_objects_name, "Nucs")
        self.assertTrue(module.wants_branchpoint_image)
        self.assertEqual(module.branchpoint_image_name, "BPImg")

    def make_workspace(self, labels, image, mask=None,
                       intensity_image=None,
                       wants_graph=False):
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        m.add_measurement(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, 1)
        m.add_measurement(cpmeas.IMAGE, cpmeas.GROUP_INDEX, 1)
        image_set = m
        img = cpi.Image(image, mask)
        image_set.add(IMAGE_NAME, img)

        object_set = cpo.ObjectSet()
        o = cpo.Objects()
        o.segmented = labels
        object_set.add_objects(o, OBJECT_NAME)

        module = M.MeasureNeurons()
        module.image_name.value = IMAGE_NAME
        module.seed_objects_name.value = OBJECT_NAME
        if intensity_image is not None:
            img = cpi.Image(intensity_image)
            image_set.add(INTENSITY_IMAGE_NAME, img)
            module.intensity_image_name.value = INTENSITY_IMAGE_NAME
        if wants_graph:
            module.wants_neuron_graph.value = True
            module.directory.dir_choice = cps.ABSOLUTE_FOLDER_NAME
            module.directory.custom_path = self.temp_dir
            module.edge_file_name.value = EDGE_FILE
            module.vertex_file_name.value = VERTEX_FILE
        module.module_num = 1

        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set, object_set,
                                  m, image_set_list)
        return workspace, module

    def test_02_01_empty(self):
        workspace, module = self.make_workspace(np.zeros((20, 10), int),
                                                np.zeros((20, 10), bool))
        #
        # Make sure module tells us about the measurements
        #
        columns = module.get_measurement_columns(None)
        features = [c[1] for c in columns]
        features.sort()
        expected = M.F_ALL
        expected.sort()
        coltypes = {}
        for feature, expected in zip(features, expected):
            expected_feature = "_".join((M.C_NEURON, expected, IMAGE_NAME))
            self.assertEqual(feature, expected_feature)
            coltypes[expected_feature] = \
                cpmeas.COLTYPE_FLOAT if expected == M.F_TOTAL_NEURITE_LENGTH \
                    else cpmeas.COLTYPE_INTEGER
        self.assertTrue(all([c[0] == OBJECT_NAME for c in columns]))
        self.assertTrue(all([c[2] == coltypes[c[1]] for c in columns]))

        categories = module.get_categories(None, OBJECT_NAME)
        self.assertEqual(len(categories), 1)
        self.assertEqual(categories[0], M.C_NEURON)
        self.assertEqual(len(module.get_categories(None, "Foo")), 0)

        measurements = module.get_measurements(None, OBJECT_NAME, M.C_NEURON)
        self.assertEqual(len(measurements), len(M.F_ALL))
        self.assertNotEqual(measurements[0], measurements[1])
        self.assertTrue(all([m in M.F_ALL for m in measurements]))

        self.assertEqual(len(module.get_measurements(None, "Foo", M.C_NEURON)), 0)
        self.assertEqual(len(module.get_measurements(None, OBJECT_NAME, "Foo")), 0)

        for feature in M.F_ALL:
            images = module.get_measurement_images(None, OBJECT_NAME,
                                                   M.C_NEURON, feature)
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0], IMAGE_NAME)

        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature in M.F_ALL:
            mname = "_".join((M.C_NEURON, expected, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 0)

    def test_02_02_trunk(self):
        '''Create an image with one soma with one neurite'''
        image = np.zeros((20, 15), bool)
        image[9, 5:] = True
        labels = np.zeros((20, 15), int)
        labels[6:12, 2:8] = 1
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, 0),
                                  (M.F_NUMBER_TRUNKS, 1)):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], expected)

    def test_02_03_trunks(self):
        '''Create an image with two soma and a neurite that goes through both'''
        image = np.zeros((30, 15), bool)
        image[1:25, 7] = True
        labels = np.zeros((30, 15), int)
        labels[6:13, 3:10] = 1
        labels[18:26, 3:10] = 2
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, [0, 0]),
                                  (M.F_NUMBER_TRUNKS, [2, 1])):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 2)
            for i in range(2):
                self.assertEqual(data[i], expected[i])

    def test_02_04_branch(self):
        '''Create an image with one soma and a neurite with a branch'''
        image = np.zeros((30, 15), bool)
        image[6:15, 7] = True
        image[15 + np.arange(3), 7 + np.arange(3)] = True
        image[15 + np.arange(3), 7 - np.arange(3)] = True
        labels = np.zeros((30, 15), int)
        labels[1:8, 3:10] = 1
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, 1),
                                  (M.F_NUMBER_TRUNKS, 1)):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], expected)

    def test_02_05_img_667(self):
        '''Create an image with a one-pixel soma and a neurite with a branch

        Regression test of IMG-667
        '''
        image = np.zeros((30, 15), bool)
        image[6:15, 7] = True
        image[15 + np.arange(3), 7 + np.arange(3)] = True
        image[15 + np.arange(3), 7 - np.arange(3)] = True
        labels = np.zeros((30, 15), int)
        labels[10, 7] = 1
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, 1),
                                  (M.F_NUMBER_TRUNKS, 2)):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], expected,
                             "%s: expected %d, got %d" % (feature, expected, data[0]))

    def test_02_06_quadrabranch(self):
        '''An odd example that I noticed and thought was worthy of a test

        You get this pattern:
              x
              I
            I   I
            I   I
            I   I
              I
            x   x

            And there should be 3 trunks (or possibly two trunks and a branch)
        '''
        image = np.zeros((30, 15), bool)
        image[6:15, 7] = True
        image[15 + np.arange(3), 7 + np.arange(3)] = True
        image[15 + np.arange(3), 7 - np.arange(3)] = True
        labels = np.zeros((30, 15), int)
        labels[13, 7] = 1
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, 0),
                                  (M.F_NUMBER_TRUNKS, 3)):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], expected,
                             "%s: expected %d, got %d" % (feature, expected, data[0]))

    def test_02_07_wrong_size(self):
        '''Regression of img-961, image and labels size differ

        Assume that image is primary, labels outside of image are ignored
        and image outside of labels is unlabeled.
        '''
        image = np.zeros((40, 15), bool)
        image[1:25, 7] = True
        labels = np.zeros((30, 20), int)
        labels[6:13, 3:10] = 1
        labels[18:26, 3:10] = 2
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        for feature, expected in ((M.F_NUMBER_NON_TRUNK_BRANCHES, [0, 0]),
                                  (M.F_NUMBER_TRUNKS, [2, 1])):
            mname = "_".join((M.C_NEURON, feature, IMAGE_NAME))
            data = m.get_current_measurement(OBJECT_NAME, mname)
            self.assertEqual(len(data), 2)
            for i in range(2):
                self.assertEqual(data[i], expected[i])

    def test_02_08_skeleton_length(self):
        #
        # Soma ends at x=8, neurite ends at x=15. Length should be 7
        #
        image = np.zeros((20, 20), bool)
        image[9, 5:15] = True
        labels = np.zeros((20, 20), int)
        labels[6:12, 2:8] = 1
        workspace, module = self.make_workspace(labels, image)
        module.run(workspace)
        m = workspace.measurements
        ftr = "_".join((M.C_NEURON, M.F_TOTAL_NEURITE_LENGTH, IMAGE_NAME))
        result = m[OBJECT_NAME, ftr]
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 5,
                               delta=np.sqrt(np.finfo(np.float32).eps))

    def read_graph_file(self, file_name):
        type_dict = dict(image_number="i4", v1="i4", v2="i4", length="i4",
                         total_intensity="f8", i="i4", j="i4",
                         vertex_number="i4", labels="i4", kind="S1")

        path = os.path.join(self.temp_dir, file_name)
        fd = open(path, "r")
        fields = fd.readline().strip().split(",")
        dt = np.dtype(dict(names=fields,
                           formats=[type_dict[x] for x in fields]))
        pos = fd.tell()
        if len(fd.readline()) == 0:
            return np.recarray(0, dt)
        fd.seek(pos)
        return np.loadtxt(fd, dt, delimiter=",")

    def test_03_00_graph(self):
        '''Does graph neurons work on an empty image?'''
        workspace, module = self.make_workspace(
                np.zeros((20, 10), int),
                np.zeros((20, 10), bool),
                intensity_image=np.zeros((20, 10)), wants_graph=True)
        module.prepare_run(workspace)
        module.run(workspace)
        edge_graph = self.read_graph_file(EDGE_FILE)
        vertex_graph = self.read_graph_file(VERTEX_FILE)
        self.assertEqual(len(edge_graph), 0)
        self.assertEqual(len(vertex_graph), 0)

    def test_03_01_graph(self):
        '''Make a simple graph'''
        #
        # The skeleton looks something like this:
        #
        #   .   .
        #    . .
        #     .
        #     .
        i, j = np.mgrid[-10:11, -10:11]
        skel = (i < 0) & (np.abs(i) == np.abs(j))
        skel[(i >= 0) & (j == 0)] = True
        #
        # Put a single label at the bottom
        #
        labels = np.zeros(skel.shape, int)
        labels[(i > 8) & (np.abs(j) < 2)] = 1
        np.random.seed(31)
        intensity = np.random.uniform(size=skel.shape)
        workspace, module = self.make_workspace(
                labels, skel, intensity_image=intensity, wants_graph=True)
        module.prepare_run(workspace)
        module.run(workspace)
        edge_graph = self.read_graph_file(EDGE_FILE)
        vertex_graph = self.read_graph_file(VERTEX_FILE)
        vidx = np.lexsort((vertex_graph["j"], vertex_graph["i"]))
        #
        # There should be two vertices at the bottom of the array - these
        # are bogus artifacts of the object hitting the edge of the image
        #
        for vidxx in vidx[-2:]:
            self.assertEqual(vertex_graph["i"][vidxx], 20)
        vidx = vidx[:-2]

        expected_vertices = ((0, 0), (0, 20), (10, 10), (17, 10))
        self.assertEqual(len(vidx), len(expected_vertices))
        for idx, v in enumerate(expected_vertices):
            vv = vertex_graph[vidx[idx]]
            self.assertEqual(vv["i"], v[0])
            self.assertEqual(vv["j"], v[1])

        #
        # Get rid of edges to the bogus vertices
        #
        for v in ("v1", "v2"):
            edge_graph = edge_graph[vertex_graph["i"][edge_graph[v] - 1] != 20]

        eidx = np.lexsort((vertex_graph["j"][edge_graph["v1"] - 1],
                           vertex_graph["i"][edge_graph["v1"] - 1],
                           vertex_graph["j"][edge_graph["v2"] - 1],
                           vertex_graph["i"][edge_graph["v2"] - 1]))
        expected_edges = (((0, 0), (10, 10), 11, np.sum(intensity[(i <= 0) & (j <= 0) & skel])),
                          ((0, 20), (10, 10), 11, np.sum(intensity[(i <= 0) & (j >= 0) & skel])),
                          ((10, 10), (17, 10), 8, np.sum(intensity[(i >= 0) & (i <= 7) & skel])))
        for i, (v1, v2, length, total_intensity) in enumerate(expected_edges):
            ee = edge_graph[eidx[i]]
            for ve, v in ((v1, ee["v1"]), (v2, ee["v2"])):
                self.assertEqual(ve[0], vertex_graph["i"][v - 1])
                self.assertEqual(ve[1], vertex_graph["j"][v - 1])
            self.assertEqual(length, ee["length"])
            self.assertAlmostEqual(total_intensity, ee["total_intensity"], 4)

    def test_03_02_four_branches(self):
        '''Test four branchpoints touching the same edge

        This exercises quite a bit of corner-case code. The permutation
        code kicks in when more than one branchpoint touches an edge's end.
        The "best edge wins" code kicks in when a branch touches another branch.
        '''
        skel = np.array(
                ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                 (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
                 (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                 (0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
                 (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
                 (0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0),
                 (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0),
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), bool)

        poi = np.array(
                ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                 (0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0),
                 (0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0),
                 (0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 4, 0, 0, 0),
                 (0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0),
                 (0, 0, -3, 0, -4, 0, 0, 0, 0, -5, 0, -6, 0, 0),
                 (0, 8, 0, 0, 0, 9, 0, 0, 10, 0, 0, 0, 11, 0),
                 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)), int)

        np.random.seed(32)
        image = np.random.uniform(size=skel.shape)
        labels = np.zeros(skel.shape, int)
        labels[-2:, -2:] = 1  # attach the object to the lower left corner

        expected_edges = ((2, 3, 2, -10),
                          (2, 4, 8, 1),
                          (2, 5, 8, 1),
                          (2, 6, 3, -1),
                          (3, 4, 8, 1),
                          (3, 5, 8, 1),
                          (3, 8, 3, -3),
                          (3, 9, 3, -4),
                          (4, 5, 2, -10),
                          (4, 7, 3, -2),
                          (5, 10, 3, -5))
        workspace, module = self.make_workspace(
                labels, skel, intensity_image=image, wants_graph=True)
        module.prepare_run(workspace)
        module.run(workspace)
        vertex_graph = self.read_graph_file(VERTEX_FILE)
        edge_graph = self.read_graph_file(EDGE_FILE)

        vertex_number = np.zeros(len(np.unique(poi[poi >= 1])), int)
        for v in vertex_graph:
            p = poi[v["i"], v["j"]]
            if p > 1:
                vertex_number[p - 2] = v["vertex_number"]
        poi_number = np.zeros(len(vertex_graph) + 1, int)
        poi_number[vertex_number] = np.arange(2, len(vertex_number) + 2)

        found_edges = [False] * len(expected_edges)
        off = -np.min([x[3] for x in expected_edges])
        for e in edge_graph:
            v1 = e["v1"]
            v2 = e["v2"]
            length = e["length"]
            total_intensity = e["total_intensity"]
            poi1 = poi_number[v1]
            poi2 = poi_number[v2]
            if poi1 == 0 or poi2 == 0:
                continue
            if poi1 > poi2:
                poi2, poi1 = (poi1, poi2)
            ee = [(i, p1, p2, l, mid) for i, (p1, p2, l, mid)
                  in enumerate(expected_edges)
                  if p1 == poi1 and p2 == poi2]
            self.assertEqual(len(ee), 1)
            i, p1, p2, l, mid = ee[0]
            self.assertEqual(l, length)
            active_poi = np.zeros(np.max(poi) + off + 1, bool)
            active_poi[np.array([poi1, poi2, mid]) + off] = True
            expected_intensity = np.sum(image[active_poi[poi + off]])
            self.assertAlmostEqual(expected_intensity, total_intensity, 4)
            found_edges[i] = True
        self.assertTrue(all(found_edges))
