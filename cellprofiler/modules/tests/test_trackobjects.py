'''test_trackobjects.py - testing of the TrackObjects module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import base64
from matplotlib.image import pil_to_array
import numpy as np
import os
import Image as PILImage
import scipy.ndimage
from StringIO import StringIO
import unittest
import zlib

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw

import cellprofiler.modules.trackobjects as T

OBJECT_NAME = "objects"

class TestTrackObjects(unittest.TestCase):
    def test_01_01_load_matlab(self):
        '''Load a Matlab pipeline with a TrackObjects module'''
        data = ('eJwBIwTc+01BVExBQiA1LjAgTUFULWZpbGUsIFBsYXRmb3JtOiBQQ1dJTiwg'
                'Q3JlYXRlZCBvbjogVGh1IEp1bCAzMCAxMzo0ODo1MCAyMDA5ICAgICAgICAg'
                'ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAAAUlN'
                'DwAAAJsDAAB4nORYW2/TMBR2LyuFjWkXQCD1IbzxMLaWCQnedkOjElumrZpA'
                '4mFu43WGNK5yqVZ+DU/8Dn4O/4EX4iZpHa+tHTfLxmbJck/q851zPtvHl0UA'
                'wNlLAEp+W/ZrHgRlLpRzTKXyCXJdbLWdOVAEz8Pvv/16Cm0MmyY6haaHHDAs'
                '0fe6dU4a/e7wrwNieCY6hB22s18OvU4T2Y5+HimGfx/hS2Se4B8IxEvU7Rj1'
                'sIOJFeqH+PzXoV3icnYX/fpnecRDjuOBtqvMd9p/C4z6F8fwtsT0XwprA126'
                'rz9cwpardaDbuqA47wQ4DzgcKus2bh/v7wAZ/TKnXw71923Yl9EvcfqlwTi1'
                'TISBlP2x/veQbcLuQL8q0C/G9ItgX9f3VO3ukk4TW0jKb1HcWwL9R5w+lfeI'
                'ZhFX85xwIt8Y/8z8EfFfiOkXgLUBle1uVtfe+yVV/kQ4DzkcKm/bCJ5cwG6U'
                'T7IcT9F6TDoeX/xclqb/Sdcj5TJL+7kYTg7UJPWum7dZ1+F950/V79tu/zFn'
                'n8q663jatgG7Lu6hjOPIx3Dy4G01W/uT9pOkerX1mlK8hyTw86NA7wkXL5Vp'
                'jnZa0EQatAwtOFdGcafFn3IeQYbyeaK6XnuzVgvsy8Yzbl/Vm9+Qf7at7zE4'
                'SdfV5iz2B/soQlYC+/y8qobzSlU/631l0nlix7+KqfJYt1xkOdjtMzxErQhv'
                'nsOjcsOGre/I8KeHM8QRrb9lDofK2DJwDxseNDXcge3hLW7WeTst3qhVzTOz'
                '+LXtucS/KOLWDH7x62s9Bb5k/FLBbUf5VSHOSeswKc6s4yi7rtP2SylPeK7p'
                '34cdkT8lDi8qEV6e0ZONR3a80uYpas/K09972Pwz6/gPklXbJl43e5xx+ZgM'
                '9ukR0DS+VfI0g6/5ORt1E/An629WOOPe8UbjEISnkv9UxyVtf9P287bzKcKR'
                'XXeT2vvGZ9I2rXiui5fr8vem7GTl///aiva3FRDnh8okODdd2eDG4f/MTT5n'
                '5MLf7DljS+DPuPz0iUCjzlyIVNf3LjGJ3SD0mUOan6ccDpXrBrJcfN4/snGH'
                'vSvI4D3j8Kh8gKDj2Sh4XmAe72XiXODwqDy4jwZoThQne37mz7v8OOVDeWV+'
                'tTwP1O5h0fuhit253EKhAK7GL9KnftH58rXyoqJXflX+VoLvFOdzYfo8PQDx'
                'efoKTO4flbvU/y62/wAAAP//42OAgFGatjQAqXa4itAqDLA=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        # Module #5: TrackObjects revision - 3
        # Choose a tracking method    Overlap
        # What did you call the objects you want to track?    Nuclei
        # What category of measurement you want to use?    AreaShape
        # What feature you want to use?  Area
        # Which image's measurements do you want to use?    OrigRGB
        # what previously measured size scale do you want to use?    1
        # Choose the neighborhood    50
        # How do you want to display the tracked objects?  Grayscale and Number
        # Select the number you want displayed    Object ID
        # Do you want to calculate statistics?    Yes
        # What do you want to call the resulting image with tracked, color-coded objects? Type "Do not use" to ignore.    TrackedObjs
        self.assertEqual(len(pipeline.modules()), 5)
        module = pipeline.modules()[4]
        self.assertTrue(isinstance(module, T.TrackObjects))
        self.assertEqual(module.tracking_method, T.TM_OVERLAP)
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.measurement.value, "AreaShape_Area_OrigRGB_1")
        self.assertEqual(module.pixel_radius.value, 50)
        self.assertEqual(module.display_type.value, "Grayscale and Number")
        self.assertTrue(module.wants_image.value)
        self.assertEqual(module.image_name.value, "TrackedObjs")
    
    def test_01_02_load_v1(self):
        '''load a version 1 pipeline'''
        data = ('eJztnFtv2zYUgOXEua1d4XUP60sB7m0YEsFO563Jy+QkTeIhjoPGaLenjJFo'
                'h50sChLl2ft1/Un9CRMtKZZZ2VLkSySHAgTn0Px4eM7hTbTCRq11UTsCVbkM'
                'GrXWXhvrCFzpkLaJ1T0EBt0FxxaCFGmAGIegQQzwh6OD/d9A5dfDX/YPKwdg'
                'v1w+kNJdhXrjhfvRVyVp0/3cdu81/6sNXy6EbiZfI0qx0bE3pKL0yk//7N4f'
                'oIXhrY4+QN1B9khFkF432qQ1MO+/ahDN0dEl7IYzu9el071Flt1sB6D/9RXu'
                'I/0a/4c4E4Js71EP25gYPu+Xz6fe6yWU08v88EUe+aHA+WHLvV+H0ln+c2mU'
                'vxjht+9C+Uu+jA0N97DmQB3gLuzc14KV9zamvC2uPCY3Ldx5f3aUiN/m+G2f'
                'P7PgIAm/yfFMvnRUHeEZ6t9Dlg7NIV+O4dfG+DXpkiTTu8HpZXKdOd/3uxLD'
                'lzie3S3Up3vv+lCloAupepfW/mPSvcUGSmRHnP8X5b9Z4x7X7uL8/4zjmdyy'
                'oPoP0pq3n7welMT+wlg5BemNNJ/6p4n7m/LugXt9Ve9Njg+ugN/xPxn3Z4ze'
                'Hzm9TK65s0lTVR0TI+3mhACDUODY6MYfR24qofKVmPKfc+UzuelQ3W3PQfdK'
                'FJfiWDlFt10aKIn+bzj9TB6ZNLLjoe2ikpBbH+PWpb/c0XyWeD60/1bL6ce/'
                'Y6ITS0o2j33P8Uw+tUjXm8EAW7MY7kSe2o5gHMpLnJQYfVHzxdDfABoa8NYi'
                'aftXA/aBSWwbu2ueoD6zrhvi4v8DxzP5BLWho1Mw7OjgBFtIpcQa5CqOafWl'
                'mi+8cdGeZb45cpfWafwrz2ivEsN9y9WXyU1qO6CmQZPiXrDOWlR/S7o+SDNO'
                'frzDFM3AU1aPCPvnuT6b2L+RliruFbkyt361bC6JfabRWeg8lXbciFpnl+XK'
                '/m7l8f36EC5uPnkljdvJ5GA+IQ41HQq0YEKZJU5KDLfD1YPJ3roeIWOCH5LE'
                'u7xi/SftfMKv68+azZNFzkNR8awbFBk2poM5+ymLcXkrPe4+UrA+UWL4pxan'
                'tOPTC2ncT0w+O706xzZ1n5Bl2IvY91lkvSvlx/fvoteN73rIGgB1oOrJn2cn'
                '7V8sq99XZ/RT1uISta9zAW0aCksm47LI59Msj2PLjsOsz9+rEg/erx1/Pzpr'
                '9RT9Ihv1FHHIRj1XKQ554ZI9t5cfvZ6ryCnS9Dgsev9YcGKcygsn4pANLsn6'
                'Qq4+fj1XkVOk6XGI/L3vXwJUHdo2in4fKMv25oU7l6bHJep9x48Id+7Y66s9'
                '9qKmoaIc2p0XTpGmxydqX+2UWKhjEcfQ8mdvHjkxv2eDW8V99bTc570RV+C4'
                'qPfelzk+DV8xZAOUufxyotYZ5PYTUumooDzFOQ/rhZB/ATY0ZM6hHkpMPZLG'
                'Oa6cZbffRfmjxOkvjen3wpKndvjU7BWc4LLQrxY9jq06p0jT/Rv1/wHE+537'
                '3sN5sjdr/l21eeGp2Ss4wYn+ITjBLaadz6ucvPhNcIITnOAEt1zuPMRFzTMv'
                'pfF5hsnh5+DwRnKe7Bac4AQnOMEJLs+cEuLE7wSCE5zgBCc4wQlOcIITnOAE'
                'NwdufcQVOI7J4XOfWP6/Q3qinst+DuUv+bKKdN20CDvz1ZK7w8PUbVknUPNO'
                'M5cv3D/roYPNk+jZ5fTsTtKjskNUKWHnXMjDA1VbhJ3BHOgxY/QonB5lkh6s'
                'IYPi9sC0XKscSrqQYlWu+6lXbmotSGV6+zF6jzi9R5P0dhG0HQt57x5Dy5Xu'
                'oInkhpfcHCazA5yvWXJy/8qcfnmS/uFhlZ52Wx6eoOnptD09dzF6qpye6iQ9'
                'ZHjo++DG36S3Ze8U+EFwSEtCuxK3Txv2kN8+r90/+fYZPk9mJ0JPuN+s+XLp'
                '+eu1ra2XU/urJI3301H//fJ7Wr3rhWLBvb7aV3kWwxd9W/lr2H4LDxs3fpIm'
                '5w9sFvmT5f8f3fzddw==')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        module = pipeline.modules()[4]
        self.assertTrue(isinstance(module, T.TrackObjects))
        self.assertEqual(module.tracking_method, T.TM_OVERLAP)
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.pixel_radius.value, 50)
        self.assertEqual(module.display_type.value, "Color and Number")
        self.assertTrue(module.wants_image.value)
        self.assertEqual(module.image_name.value, "TrackedObjs")
    
    def test_01_03_load_v2(self):
        data = ('eJztWnFv2kYUPwjJlnWaMk1T90+l+7PZgmWzsrXRlEJh3dAKQQW1qqquvdhH'
                'uO3wIftMwqZK/Qj7OPtY/QjzGRvMxYmNSYB2tmKZ93y/93v37t3z+eJmtfuk'
                '+giWFRU2q91ij1AM2xTxHrMGh9DkB7BmYcSxAZl5CJvMhC02gtoP7t+hVjrU'
                'yrCkqg9AuiPXaH7hXv5RANhxr5+6Z96/te3LudAp5A7mnJin9jYogG98/Xv3'
                'fIYsgk4ofoaog+0ZRaBvmD3WHQ+nt5rMcChuoUG4sXu0nMEJtuzjXgD0b7fJ'
                'OaYd8heWuhA0e4pHxCbM9PG+fVk75WVc4u302dljy3VHsv8Icb3f4e4IzOtF'
                '3N59PYtbTorblnveCelF+1/BrH0hIs5fhtrv+TIxDTIihoMoJAN0OvVa2FNj'
                '7G3N2dsC9VbVw92PwX0i+SHk4xG2KBp6+EoMfk/Ci7OLz3nx53OkczgQIU3i'
                'x45kR8gtR6eYgET9iMPHxS83h8+B70Ey3m2JV8iaenBPBcl4C3P4AmgxEyeJ'
                '+2cSr5DrDJqMQ8f28zdN3rxwsy4JLj+Hy4NSORmfjGux5XBxcYrKzxqjzILI'
                'NOCkViSZr7clO0Ku4x5yKIcNMVlhnVhY58warzTuUfHbkXDBEeB2/WuS+N2S'
                '+i3kY2478BfKThBNbOdzyY6QuxbS/8RGDVNqB3ZuKm4yTlPUdDg1HS7wM81z'
                'QVVU7zjQ/B/+/Zv0P+08leuZ67u2TH7eVF7vgvk4C7lhcmzahI+vgT/uuRFe'
                'R+z5cq2PTBPT0ib4n3a9sWheaerN+ik/17WUuB8jcNfp52XrgOuoh4v4+S6G'
                '7zcwn3dC/v3uw/ZP4kUGHynf7b8W0nO3pj9lZ0cvq8X2q/1A4z54nYF59FIt'
                'Pnj1t3ZQejtp3CEu0lPuT/2oxPiRZv2zSBz6Mfz3JX4hi768wMjyO3jv7X5R'
                'qNwXOd73dSVfV0fjmWYd47xEfbpQz1fhbyWGL6oe1sacDSmyByE7q/Z70TpT'
                'SumnGrGe2YT6lAT3IdanVa/TP/Z6tKp1iqqU1+JnJcbPqPes7hmDulu/bH8n'
                'aB1+p3lfeY7JaV9sY47Ehp2p45C9TYt71DriMbPwqcUc01if3x/L/JOfb+Ul'
                '+f79arF92FXmjbdpKxJnuDz/KucrO/kD69xzHBLTwMMF4hBVt0L21haHDJfh'
                'NhlXAVfPq6h961l9mUzTD6m/GS7DbTKuArL5mOGy/Mlw2XhuMq4Crh6XTX0v'
                'y3AZLsNluP87bpib4eT9O3lfU7R/E+KJqvffgvl6L2QdUzq0mPi+1lIG3keg'
                'tkIZMiZfVSpP3J+N0AeWnl8xPBWJp3IZDzGwyUlvPLRcNoezAeJEVxq+tu1q'
                'q4E2Sf8UiVe5jJeLj6kmO2+24n1ZdTwRLo7XbgRPOO55V7p9Z/fKcZbHdzbu'
                '7x+m4SsU8hf+L30rBlcI+SQOb38YLJZfd69oH/Rxle0XjVsul1u63zOewtSn'
                'if3VtP8PaZbd5A==')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))        
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, T.TrackObjects))
        self.assertEqual(module.tracking_method, T.TM_OVERLAP)
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.pixel_radius.value, 25)
        self.assertEqual(module.display_type.value, "Color and Number")
        self.assertFalse(module.wants_image)
        
    def test_01_04_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9227

TrackObjects:[module_num:1|svn_version:\'9227\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    Choose a tracking method:LAP
    Select the objects to track:Nuclei
    Select measurement to use:AreaShape_Area
    Select pixel distance:80
    Select display option:Color and Number
    Save color-coded image?:No
    Name the output image:TrackedCells
    Cost of being born:100
    Cost of dying:100
    Do you want to run the second phase of the LAP algorithm?:Yes
    Gap cost\x3A:40
    Split alternative cost\x3A:41
    Merge alternative cost\x3A:42
    Maximum gap displacement\x3A:53
    Maximum split score\x3A:54
    Maximum merge score\x3A:55
    Maximum gap:6
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, T.TrackObjects))
        self.assertEqual(module.tracking_method, T.TM_LAP)
        self.assertEqual(module.object_name.value, "Nuclei")
        self.assertEqual(module.pixel_radius.value, 80)
        self.assertEqual(module.display_type.value, "Color and Number")
        self.assertFalse(module.wants_image)
        self.assertEqual(module.measurement, "AreaShape_Area")
        self.assertEqual(module.image_name, "TrackedCells")
        self.assertEqual(module.born_cost, 100)
        self.assertEqual(module.die_cost, 100)
        self.assertTrue(module.wants_second_phase)
        self.assertEqual(module.split_cost, 41)
        self.assertEqual(module.merge_cost, 42)
        self.assertEqual(module.max_gap_score, 53)
        self.assertEqual(module.max_split_score, 54)
        self.assertEqual(module.max_merge_score, 55)
        self.assertEqual(module.max_frame_distance, 6)
                         
        
    def runTrackObjects(self, labels_list, fn = None, measurement = None):
        '''Run two cycles of TrackObjects
        
        labels1 - the labels matrix for the first cycle
        labels2 - the labels matrix for the second cycle
        fn - a callback function called with the module and workspace. It has
             the signature, fn(module, workspace, n) where n is 0 when
             called prior to prepare_run, 1 prior to first iteration
             and 2 prior to second iteration.
        
        returns the measurements
        '''
        module = T.TrackObjects()
        module.module_num = 1
        module.object_name.value = OBJECT_NAME
        module.pixel_radius.value = 50
        module.measurement.value = "measurement"
        measurements = cpmeas.Measurements()
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()

        if fn:
            fn(module, None, 0)
        module.prepare_run(pipeline, image_set_list, None)
        
        first = True
        for labels, index in zip(labels_list, range(len(labels_list))):
            object_set = cpo.ObjectSet()
            objects = cpo.Objects()
            objects.segmented = labels
            object_set.add_objects(objects, OBJECT_NAME)
            image_set = image_set_list.get_image_set(index)
            if first:
                first = False
            else:
                measurements.next_image_set()
            if measurement is not None:
                measurements.add_measurement(OBJECT_NAME, "measurement",
                                             np.array(measurement[index]))
            workspace = cpw.Workspace(pipeline, module, image_set,
                                      object_set, measurements, image_set_list)
            if fn:
                fn(module, workspace, index+1)
            
            module.run(workspace)
        return measurements
    
    def test_02_01_track_nothing(self):
        '''Run TrackObjects on an empty labels matrix'''
        columns = []
        def fn(module, workspace, index, columns = columns):
            if workspace is not None and index == 0:
                columns += module.get_measurement_columns(workspace.pipeline)
        
        measurements = self.runTrackObjects((np.zeros((10,10),int),
                                             np.zeros((10,10),int)), fn)
        
        features = [ feature 
                     for feature in measurements.get_feature_names(OBJECT_NAME)
                     if feature.startswith(T.F_PREFIX)]
        self.assertTrue(all([column[1] in features
                             for column in columns
                             if column[0] == OBJECT_NAME]))
        for feature in T.F_ALL:
            name = "_".join((T.F_PREFIX, feature, "50"))
            self.assertTrue(name in features)
            value = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(value), 0)
        
        features = [ feature for feature in measurements.get_feature_names(cpmeas.IMAGE)
                     if feature.startswith(T.F_PREFIX)]
        self.assertTrue(all([column[1] in features
                             for column in columns
                             if column[0] == cpmeas.IMAGE]))
        for feature in T.F_IMAGE_ALL:
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "50"))
            self.assertTrue(name in features)
            value = measurements.get_current_image_measurement(name)
            self.assertEqual(value, 0)
            
    def test_02_01_00_track_one_then_nothing(self):
        '''Run track objects on an object that disappears

        Regression test of IMG-1090
        '''
        labels = np.zeros((10,10),int)
        labels[3:6, 2:7] = 1
        measurements = self.runTrackObjects((labels,
                                             np.zeros((10,10),int)))
        feature = "_".join((T.F_PREFIX, T.F_LOST_OBJECT_COUNT, 
                            OBJECT_NAME, "50"))
        value = measurements.get_current_image_measurement(feature)
        self.assertEqual(value, 1)
    
    def test_02_02_track_one_distance(self):
        '''Track an object that doesn't move using distance'''
        labels = np.zeros((10,10),int)
        labels[3:6, 2:7] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 1
                module.tracking_method.value = T.TM_DISTANCE
        measurements = self.runTrackObjects((labels, labels), fn)
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "1"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertAlmostEqual(m(T.F_TRAJECTORY_X), 0)
        self.assertAlmostEqual(m(T.F_TRAJECTORY_Y), 0)
        self.assertAlmostEqual(m(T.F_DISTANCE_TRAVELED), 0)
        self.assertAlmostEqual(m(T.F_INTEGRATED_DISTANCE), 0)
        self.assertEqual(m(T.F_LABEL), 1)
        self.assertEqual(m(T.F_PARENT), 1)
        self.assertEqual(m(T.F_LIFETIME), 1)
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "1"))
            return measurements.get_current_image_measurement(name)
        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)
    
    def test_02_03_track_one_moving(self):
        '''Track an object that moves'''
        
        labels_list =  []
        distance = 0
        last_i, last_j = (0,0)
        for i_off, j_off in ((0,0),(2,0),(2,1),(0,1)):
            distance = i_off - last_i + j_off - last_j
            last_i, last_j = (i_off, j_off)
            labels = np.zeros((10,10),int)
            labels[4+i_off:7+i_off,4+j_off:7+j_off] = 1
            labels_list.append(labels)
        
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 3
                module.tracking_method.value = T.TM_DISTANCE
        measurements = self.runTrackObjects(labels_list, fn)
        def m(feature, expected):
            name = "_".join((T.F_PREFIX, feature, "3"))
            value_set = measurements.get_all_measurements(OBJECT_NAME, name)
            self.assertEqual(len(expected), len(value_set))
            for values, x in zip(value_set,expected):
                self.assertEqual(len(values), 1)
                self.assertAlmostEqual(values[0], x)
        
        m(T.F_TRAJECTORY_X, [0,0,1,0])
        m(T.F_TRAJECTORY_Y, [0,2,0,-2])
        m(T.F_DISTANCE_TRAVELED, [0,2,1,2])
        m(T.F_INTEGRATED_DISTANCE, [0,2,3,5])
        m(T.F_LABEL, [1,1,1,1])
        m(T.F_LIFETIME, [0,1,2,3])
        m(T.F_LINEARITY, [1,1,np.sqrt(5)/3,1.0/5.0])
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "3"))
            return measurements.get_current_image_measurement(name)
        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)
    
    def test_02_04_track_split(self):
        '''Track an object that splits'''
        labels1 = np.zeros((10,10), int)
        labels1[1:9,1:9] = 1
        labels2 = np.zeros((10,10), int)
        labels2[1:5,1:9] = 1
        labels2[5:9,1:9] = 2
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 5
                module.tracking_method.value = T.TM_DISTANCE
        measurements = self.runTrackObjects((labels1,labels2, labels2), fn)
        def m(feature, idx):
            name = "_".join((T.F_PREFIX, feature, "5"))
            values = measurements.get_all_measurements(OBJECT_NAME, name)[idx]
            self.assertEqual(len(values), 2)
            return values

        labels = list(m(T.F_LABEL,2))
        self.assertEqual(len(labels),2)
        self.assertTrue(2 in labels)
        self.assertTrue(3 in labels)
        parents = m(T.F_PARENT,1)
        self.assertTrue(np.all(parents == 1))
        parents = m(T.F_PARENT,2)
        self.assertTrue(np.all(parents == labels))
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "5"))
            return measurements.get_all_measurements(cpmeas.IMAGE, name)[1]
        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 0)
        self.assertEqual(m(T.F_SPLIT_COUNT), 1)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)
    
    def test_02_05_track_negative(self):
        '''Track unrelated objects'''
        labels1 = np.zeros((10,10), int)
        labels1[1:5,1:5] = 1
        labels2 = np.zeros((10,10), int)
        labels2[6:9,6:9] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 1
                module.tracking_method.value = T.TM_DISTANCE
        measurements = self.runTrackObjects((labels1,labels2), fn)
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "1"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT), 0)
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, OBJECT_NAME, "1"))
            return measurements.get_current_image_measurement(name)
        self.assertEqual(m(T.F_NEW_OBJECT_COUNT), 1)
        self.assertEqual(m(T.F_LOST_OBJECT_COUNT), 1)
        self.assertEqual(m(T.F_SPLIT_COUNT), 0)
        self.assertEqual(m(T.F_MERGE_COUNT), 0)

    def test_02_06_track_ambiguous(self):
        '''Track disambiguation from among two possible parents'''
        labels1 = np.zeros((20,20), int)
        labels1[1:4,1:4] = 1
        labels1[16:19,16:19] = 2
        labels2 = np.zeros((20,20), int)
        labels2[10:15,10:15] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 20
                module.tracking_method.value = T.TM_DISTANCE
        measurements = self.runTrackObjects((labels1,labels2), fn)
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "20"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT), 2)

    def test_03_01_overlap_positive(self):
        '''Track overlapping objects'''
        labels1 = np.zeros((10,10), int)
        labels1[3:6,4:7] = 1
        labels2 = np.zeros((10,10), int)
        labels2[4:7,5:9] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = T.TM_OVERLAP
        measurements = self.runTrackObjects((labels1,labels2), fn)
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertEqual(m(T.F_LABEL), 1)
        self.assertEqual(m(T.F_PARENT), 1)
    
    def test_03_02_overlap_negative(self):
        '''Track objects that don't overlap'''
        labels1 = np.zeros((20,20), int)
        labels1[3:6,4:7] = 1
        labels2 = np.zeros((20,20), int)
        labels2[14:17,15:19] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = T.TM_OVERLAP
        measurements = self.runTrackObjects((labels1,labels2), fn)
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT), 0)
    
    def test_03_03_overlap_ambiguous(self):
        '''Track an object that overlaps two parents'''
        labels1 = np.zeros((20,20), int)
        labels1[1:5,1:5] = 1
        labels1[15:19,15:19] = 2
        labels2 = np.zeros((20,20), int)
        labels2[4:18,4:18] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = T.TM_OVERLAP
        measurements = self.runTrackObjects((labels1,labels2), fn)
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT), 2)
    
    def test_04_01_measurement_positive(self):
        '''Test tracking an object by measurement'''
        labels1 = np.zeros((10,10), int)
        labels1[3:6,4:7] = 1
        labels2 = np.zeros((10,10), int)
        labels2[4:7,5:9] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = T.TM_MEASUREMENTS
        measurements = self.runTrackObjects((labels1,labels2), fn, [[1],[1]])
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertEqual(m(T.F_LABEL), 1)
        self.assertEqual(m(T.F_PARENT), 1)
    
    def test_04_02_measurement_negative(self):
        '''Test tracking with too great a jump between successive images'''
        labels1 = np.zeros((20,20), int)
        labels1[3:6,4:7] = 1
        labels2 = np.zeros((20,20), int)
        labels2[14:17,15:19] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 2
                module.tracking_method.value = T.TM_MEASUREMENTS
        measurements = self.runTrackObjects((labels1,labels2), fn, [[1],[1]])
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "2"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT), 0)
    
    def test_04_03_ambiguous(self):
        '''Test measurement with ambiguous parent choice'''
        labels1 = np.zeros((20,20), int)
        labels1[1:5,1:5] = 1
        labels1[15:19,15:19] = 2
        labels2 = np.zeros((20,20), int)
        labels2[6:14,6:14] = 1
        def fn(module, workspace, idx):
            if idx == 0:
                module.pixel_radius.value = 4
                module.tracking_method.value = T.TM_MEASUREMENTS
        measurements = self.runTrackObjects((labels1,labels2), fn,[[1,10],[9]])
        def m(feature):
            name = "_".join((T.F_PREFIX, feature, "4"))
            values = measurements.get_current_measurement(OBJECT_NAME, name)
            self.assertEqual(len(values), 1)
            return values[0]
        self.assertEqual(m(T.F_LABEL), 2)
        self.assertEqual(m(T.F_PARENT), 2)
        
    def test_05_01_measurement_columns(self):
        '''Test get_measurement_columns function'''
        module = T.TrackObjects()
        module.object_name.value = OBJECT_NAME
        module.pixel_radius.value = 10
        columns = module.get_measurement_columns(None)
        self.assertEqual(len(columns), len(T.F_ALL) + len(T.F_IMAGE_ALL))
        for object_name, features in ((OBJECT_NAME, T.F_ALL),
                                      (cpmeas.IMAGE, T.F_IMAGE_ALL)):
            for feature in features:
                if object_name == OBJECT_NAME:
                    name = "_".join((T.F_PREFIX, feature, "10"))
                else:
                    name = "_".join((T.F_PREFIX, feature, 
                                     OBJECT_NAME, "10"))
                index = [column[1] for column in columns].index(name)
                self.assertTrue(index != -1)
                column = columns[index]
                self.assertEqual(column[0], object_name)
    
    def test_06_01_measurements(self):
        '''Test the different measurement pieces'''
        module = T.TrackObjects()
        module.object_name.value = OBJECT_NAME
        module.image_name.value = "image"
        module.pixel_radius.value = 10
        categories = module.get_categories(None,"Foo")
        self.assertEqual(len(categories),0)
        categories = module.get_categories(None,OBJECT_NAME)
        self.assertEqual(len(categories),1)
        self.assertEqual(categories[0], T.F_PREFIX)
        features = module.get_measurements(None, OBJECT_NAME, "Foo")
        self.assertEqual(len(features), 0)
        features = module.get_measurements(None, OBJECT_NAME, T.F_PREFIX)
        self.assertEqual(len(features), len(T.F_ALL))
        self.assertTrue(all([feature in T.F_ALL for feature in features]))
        scales = module.get_measurement_scales(None, OBJECT_NAME,
                                               T.F_PREFIX, "Foo","image")
        self.assertEqual(len(scales), 0)
        for feature in T.F_ALL:
            scales = module.get_measurement_scales(None, OBJECT_NAME,
                                                   T.F_PREFIX, feature,"image")
            self.assertEqual(len(scales), 1)
            self.assertEqual(int(scales[0]), 10)

    def make_lap2_workspace(self, objs, nimages):
        '''Make a workspace to test the second half of LAP
        
        objs - a N x 5 array of "objects" composed of the
               following pieces per object
               objs[0] - image set # for object
               objs[1] - label for object
               objs[2] - x coordinate for object
               objs[3] - y coordinate for object
               objs[4] - area for object
        '''
        module = T.TrackObjects()
        module.module_num = 1
        module.object_name.value = OBJECT_NAME
        module.tracking_method.value = T.TM_LAP
        module.wants_second_phase.value = True
        module.pixel_radius.value = 50
        
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        
        m = cpmeas.Measurements()
        for index, feature in (
            (1, module.measurement_name(T.F_LABEL)),
            (2, T.M_LOCATION_CENTER_X),
            (3, T.M_LOCATION_CENTER_Y),
            (4, module.measurement_name(T.F_AREA))):
            values = [objs[objs[:,0] == i, index] for i in range(nimages)]
            m.add_all_measurements(OBJECT_NAME, feature, values)
        m.add_all_measurements(cpmeas.IMAGE, "ImageNumber", list(range(nimages)))
        m.image_set_number = nimages
        
        image_set_list = cpi.ImageSetList()
        for i in range(nimages):
            image_set = image_set_list.get_image_set(i)
        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  m, image_set_list)
        return workspace, module
    
    def test_07_01_lap_none(self):
        '''Run the second part of LAP on one image of nothing'''
        workspace, module = self.make_lap2_workspace(np.zeros((0,5)), 1)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.post_group(workspace, np.arange(1))
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(labels[0]), 0)
        
    def test_07_02_lap_one(self):
        '''Run the second part of LAP on one image of one object'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 100, 100, 25]]), 1)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(labels[0]), 1)
        self.assertEqual(labels[0][0], 1)
        
    def test_07_03_bridge_gap(self):
        '''Bridge a gap of zero frames between two objects'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 25],
                      [1, 2, 100, 100, 25]]), 2)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The cost of bridging the gap should be 141. We set the alternative
        # score to 142 so that bridging wins.
        #
        module.gap_cost.value = 142
        module.max_gap_score.value = 142
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(labels[0]), 1)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(len(labels[1]), 1)
        self.assertEqual(labels[1][0], 1)
        
    def test_07_04_maintain_gap(self):
        '''Maintain object identity across a large gap'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 25],
                      [1, 2, 100, 100, 25]]), 2)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The cost of creating the gap should be 140 and the cost of
        # bridging the gap should be 141.
        #
        module.gap_cost.value = 140
        module.max_gap_score.value = 142
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(labels[0]), 1)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(len(labels[1]), 1)
        self.assertEqual(labels[1][0], 2)
        
    def test_07_05_filter_gap(self):
        '''Filter a gap due to an unreasonable score'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 0, 0, 25],
                      [1, 2, 100, 100, 25]]), 2)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The cost of creating the gap should be 142 and the cost of
        # bridging the gap should be 141. However, the gap should be filtered
        # by the max score
        #
        module.gap_cost.value = 142
        module.max_gap_score.value = 140
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(labels[0]), 1)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(len(labels[1]), 1)
        self.assertEqual(labels[1][0], 2)
        
    def test_07_06_split(self):
        '''Track an object splitting'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 100, 100, 50],
                      [1, 1, 110, 110, 25],
                      [1, 2, 90,   90, 25],
                      [2, 1, 110, 110, 25],
                      [2, 2, 90,   90, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The split score should be between 28 and 30.  Set the split
        # alternative cost to 15 so that the split is favored.
        #
        module.split_cost.value = 30
        module.max_split_score.value = 30
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0]), 1)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(len(labels[1]), 2)
        self.assertEqual(labels[1][0], 1)
        self.assertEqual(labels[1][1], 1)
        
    def test_07_07_dont_split(self):
        '''Track an object splitting'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 100, 100, 50],
                      [1, 1, 110, 110, 25],
                      [1, 2, 90,   90, 25],
                      [2, 1, 110, 110, 25],
                      [2, 2, 90,   90, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.split_cost.value = 28
        module.max_split_score.value = 30
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0]), 1)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(len(labels[1]), 2)
        self.assertEqual(labels[1][0], 1)
        self.assertEqual(labels[1][1], 2)

    def test_07_08_split_filter(self):
        '''Prevent a split by setting the filter too low'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 100, 100, 50],
                      [1, 1, 110, 110, 25],
                      [1, 2, 90,   90, 25],
                      [2, 1, 110, 110, 25],
                      [2, 2, 90,   90, 25]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.split_cost.value = 30
        module.max_split_score.value = 28
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0]), 1)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(len(labels[1]), 2)
        self.assertEqual(labels[1][0], 1)
        self.assertEqual(labels[1][1], 2)
        
    def test_07_09_merge(self):
        '''Merge two objects into one'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 110, 110, 25],
                      [0, 2, 90,   90, 25],
                      [1, 1, 110, 110, 25],
                      [1, 2, 90,   90, 25],
                      [2, 1, 100, 100, 50]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        module.merge_cost.value = 30
        module.max_merge_score.value = 30
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0]), 2)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(labels[0][1], 1)
        self.assertEqual(len(labels[1]), 2)
        self.assertEqual(labels[1][0], 1)
        self.assertEqual(labels[1][1], 1)
        self.assertEqual(len(labels[2]), 1)
        self.assertEqual(labels[2][0], 1)

    def test_07_10_dont_merge(self):
        '''Don't merge because of low alternative merge cost'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 110, 110, 25],
                      [0, 2, 90,   90, 25],
                      [1, 1, 110, 110, 25],
                      [1, 2, 90,   90, 25],
                      [2, 1, 100, 100, 50]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The cost of the merge is 2x 10x sqrt(2) which is between 28 and 29
        #
        module.merge_cost.value = 28
        module.max_merge_score.value = 30
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0]), 2)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(labels[0][1], 2)
        self.assertEqual(len(labels[1]), 2)
        self.assertEqual(labels[1][0], 1)
        self.assertEqual(labels[1][1], 2)
        self.assertEqual(len(labels[2]), 1)
        self.assertEqual(labels[2][0], 1)

    def test_07_11_filter_merge(self):
        '''Don't merge because of low alternative merge cost'''
        workspace, module = self.make_lap2_workspace(
            np.array([[0, 1, 110, 110, 25],
                      [0, 2, 90,   90, 25],
                      [1, 1, 110, 110, 25],
                      [1, 2, 90,   90, 25],
                      [2, 1, 100, 100, 50]]), 3)
        self.assertTrue(isinstance(module, T.TrackObjects))
        #
        # The cost of the merge is 2x 10x sqrt(2) which is between 28 and 29
        #
        module.merge_cost.value = 30
        module.max_merge_score.value = 28
        module.run_as_data_tool(workspace)
        labels = workspace.measurements.get_all_measurements(
            OBJECT_NAME, module.measurement_name(T.F_LABEL))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len(labels[0]), 2)
        self.assertEqual(labels[0][0], 1)
        self.assertEqual(labels[0][1], 2)
        self.assertEqual(len(labels[1]), 2)
        self.assertEqual(labels[1][0], 1)
        self.assertEqual(labels[1][1], 2)
        self.assertEqual(len(labels[2]), 1)
        self.assertEqual(labels[2][0], 1)
                
    def test_08_01_save_image(self):
        module = T.TrackObjects()
        module.module_num = 1
        module.object_name.value = OBJECT_NAME
        module.pixel_radius.value = 50
        module.wants_image.value = True
        module.image_name.value = "outimage"
        measurements = cpmeas.Measurements()
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()

        module.prepare_run(pipeline, image_set_list, None)
        
        first = True
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = np.zeros((640,480), int)
        object_set.add_objects(objects, OBJECT_NAME)
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, measurements, image_set_list)
        module.run(workspace)
        image = workspace.image_set.get_image(module.image_name.value)
        shape = image.pixel_data.shape
        self.assertEqual(shape[0], 640)
        self.assertEqual(shape[1], 480)
        
    
