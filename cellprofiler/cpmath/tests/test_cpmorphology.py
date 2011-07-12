"""test_cpmorphology - test the functions in cellprofiler.cpmath.cpmorphology

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"

import base64
import unittest
import numpy as np
import scipy.ndimage as scind
import scipy.misc
import scipy.io.matlab

import cellprofiler.cpmath.cpmorphology as morph
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.filter import permutations

class TestFillLabeledHoles(unittest.TestCase):
    def test_01_00_zeros(self):
        """A label matrix of all zeros has no hole"""
        image = np.zeros((10,10),dtype=int)
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output==0))
    
    def test_01_01_ones(self):
        """Regression test - an image of all ones"""
        image = np.ones((10,10),dtype=int)
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output==1))

    def test_02_object_without_holes(self):
        """The label matrix of a single object without holes has no hole"""
        image = np.zeros((10,10),dtype=int)
        image[3:6,3:6] = 1
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output==image))
    
    def test_03_object_with_hole(self):
        image = np.zeros((20,20),dtype=int)
        image[5:15,5:15] = 1
        image[8:12,8:12] = 0
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output[8:12,8:12] == 1))
        output[8:12,8:12] = 0 # unfill the hole again
        self.assertTrue(np.all(output==image))
    
    def test_04_holes_on_edges_are_not_holes(self):
        image = np.zeros((40,40),dtype=int)
        objects = (((15,25),(0,10),(18,22),(0,3)),
                   ((0,10),(15,25),(0,3),(18,22)),
                   ((15,25),(30,39),(18,22),(36,39)),
                   ((30,39),(15,25),(36,39),(18,22)))
        for idx,x in zip(range(1,len(objects)+1),objects):
            image[x[0][0]:x[0][1],x[1][0]:x[1][1]] = idx
            image[x[2][0]:x[2][1],x[3][0]:x[3][1]] = 0
        output = morph.fill_labeled_holes(image)
        for x in objects:
            self.assertTrue(np.all(output[x[2][0]:x[2][1],x[3][0]:x[3][1]]==0))
            output[x[2][0]:x[2][1],x[3][0]:x[3][1]] = 1
            self.assertTrue(np.all(output[x[0][0]:x[0][1],x[1][0]:x[1][1]]!=0))
            
    def test_05_lots_of_objects_with_holes(self):
        image = np.ones((1020,1020),bool)
        for i in range(0,51):
            image[i*20:i*20+10,:] = ~image[i*20:i*20+10,:]
            image[:,i*20:i*20+10] = ~ image[:,i*20:i*20+10]
        image = scind.binary_erosion(image, iterations = 2)
        erosion = scind.binary_erosion(image, iterations = 2)
        image = image & ~ erosion
        labeled_image,nobjects = scind.label(image)
        output = morph.fill_labeled_holes(labeled_image)
        self.assertTrue(np.all(output[erosion] > 0))
    
    def test_06_regression_diamond(self):
        """Check filling the center of a diamond"""
        image = np.zeros((5,5),int)
        image[1,2]=1
        image[2,1]=1
        image[2,3]=1
        image[3,2]=1
        output = morph.fill_labeled_holes(image)
        where = np.argwhere(image != output)
        self.assertEqual(len(where),1)
        self.assertEqual(where[0][0],2)
        self.assertEqual(where[0][1],2)
    
    def test_07_regression_nearby_holes(self):
        """Check filling an object with three holes"""
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,0,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0]])
        expec = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0]])
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output==expec))
        
    def test_08_fill_small_holes(self):
        """Check filling only the small holes"""
        image = np.zeros((10,20), int)
        image[1:-1,1:-1] = 1
        image[3:8,4:7] = 0     # A hole with area of 5*3 = 15 and not filled
        expected = image.copy()
        image[3:5, 11:18] = 0  # A hole with area 2*7 = 14 is filled
        
        def small_hole_fn(area, is_foreground):
            return area <= 14
        output = morph.fill_labeled_holes(image, size_fn = small_hole_fn)
        self.assertTrue(np.all(output == expected))
        
    def test_09_fill_binary_image(self):
        """Make sure that we can fill a binary image too"""
        image = np.zeros((10,20), bool)
        image[1:-1, 1:-1] = True
        image[3:8, 4:7] = False # A hole with area of 5*3 = 15 and not filled
        expected = image.copy()
        image[3:5, 11:18] = False # A hole with area 2*7 = 14 is filled
        def small_hole_fn(area, is_foreground):
            return area <= 14
        output = morph.fill_labeled_holes(image, size_fn = small_hole_fn)
        self.assertEqual(image.dtype.kind, output.dtype.kind)
        self.assertTrue(np.all(output == expected))
        
    def test_10_fill_bullseye(self):
        i,j = np.mgrid[-50:50, -50:50]
        bullseye = i * i + j * j < 2000
        bullseye[i * i + j * j < 1000 ] = False
        bullseye[i * i + j * j < 500 ] = True
        bullseye[i * i + j * j < 250 ] = False
        bullseye[i * i + j * j < 100 ] = True
        labels, count = scind.label(bullseye)
        result = morph.fill_labeled_holes(labels)
        self.assertTrue(np.all(result[result != 0] == bullseye[6, 43]))
        
    def test_11_dont_fill_if_touches_2(self):
        labels = np.array([
            [ 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 2, 2, 2, 0 ],
            [ 0, 1, 1, 0, 0, 2, 2, 0 ],
            [ 0, 1, 1, 1, 2, 2, 2, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0 ]])
        result = morph.fill_labeled_holes(labels)
        self

class TestAdjacent(unittest.TestCase):
    def test_00_00_zeros(self):
        result = morph.adjacent(np.zeros((10,10), int))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_one(self):
        image = np.zeros((10,10), int)
        image[2:5,3:8] = 1
        result = morph.adjacent(image)
        self.assertTrue(np.all(result==False))
        
    def test_01_02_not_adjacent(self):
        image = np.zeros((10,10), int)
        image[2:5,3:8] = 1
        image[6:8,3:8] = 2
        result = morph.adjacent(image)
        self.assertTrue(np.all(result==False))

    def test_01_03_adjacent(self):
        image = np.zeros((10,10), int)
        image[2:8,3:5] = 1
        image[2:8,5:8] = 2
        expected = np.zeros((10,10), bool)
        expected[2:8,4:6] = True
        result = morph.adjacent(image)
        self.assertTrue(np.all(result==expected))
        
    def test_02_01_127_objects(self):
        '''Test that adjacency works for int8 and 127 labels
        
        Regression test of img-1099. Adjacent sets the background to the
        maximum value of the labels matrix + 1. For 127 and int8, it wraps
        around and uses -127.
        '''
        # Create 127 labels
        labels = np.zeros((32,16), np.int8)
        i,j = np.mgrid[0:32, 0:16]
        mask = (i % 2 > 0) & (j % 2 > 0)
        labels[mask] = np.arange(np.sum(mask))
        result = morph.adjacent(labels)
        self.assertTrue(np.all(result == False))
        
class TestStrelDisk(unittest.TestCase):
    """Test cellprofiler.cpmath.cpmorphology.strel_disk"""
    
    def test_01_radius2(self):
        """Test strel_disk with a radius of 2"""
        x = morph.strel_disk(2)
        self.assertTrue(x.shape[0], 5)
        self.assertTrue(x.shape[1], 5)
        y = [0,0,1,0,0,
             0,1,1,1,0,
             1,1,1,1,1,
             0,1,1,1,0,
             0,0,1,0,0]
        ya = np.array(y,dtype=float).reshape((5,5))
        self.assertTrue(np.all(x==ya))
    
    def test_02_radius2_point_5(self):
        """Test strel_disk with a radius of 2.5"""
        x = morph.strel_disk(2.5)
        self.assertTrue(x.shape[0], 5)
        self.assertTrue(x.shape[1], 5)
        y = [0,1,1,1,0,
             1,1,1,1,1,
             1,1,1,1,1,
             1,1,1,1,1,
             0,1,1,1,0]
        ya = np.array(y,dtype=float).reshape((5,5))
        self.assertTrue(np.all(x==ya))

class TestBinaryShrink(unittest.TestCase):
    def test_01_zeros(self):
        """Shrink an empty array to itself"""
        input = np.zeros((10,10),dtype=bool)
        result = morph.binary_shrink(input,1)
        self.assertTrue(np.all(input==result))
    
    def test_02_cross(self):
        """Shrink a cross to a single point"""
        input = np.zeros((9,9),dtype=bool)
        input[4,:]=True
        input[:,4]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_03_x(self):
        input = np.zeros((9,9),dtype=bool)
        x,y = np.mgrid[-4:5,-4:5]
        input[x==y]=True
        input[x==-y]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_04_block(self):
        """A block should shrink to a point"""
        input = np.zeros((9,9), dtype=bool)
        input[3:6,3:6]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_05_hole(self):
        """A hole in a block should shrink to a ring"""
        input = np.zeros((19,19), dtype=bool)
        input[5:15,5:15]=True
        input[9,9]=False
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        self.assertTrue(len(where) > 1)
        self.assertFalse(result[9:9])

    def test_06_random_filled(self):
        """Shrink random blobs
        
        If you label a random binary image, then fill the holes,
        then shrink the result, each blob should shrink to a point
        """
        np.random.seed(0)
        input = np.random.uniform(size=(300,300)) > .8
        labels,nlabels = scind.label(input,np.ones((3,3),bool))
        filled_labels = morph.fill_labeled_holes(labels)
        input = filled_labels > 0
        result = morph.binary_shrink(input)
        my_sum = scind.sum(result.astype(int),filled_labels,np.array(range(nlabels+1),dtype=np.int32))
        my_sum = np.array(my_sum)
        self.assertTrue(np.all(my_sum[1:] == 1))
        
    def test_07_all_patterns_of_3x3(self):
        '''Run all patterns of 3x3 with a 1 in the middle
        
        All of these patterns should shrink to a single pixel since
        all are 8-connected and there are no holes
        '''
        for i in range(512):
            a = morph.pattern_of(i)
            if a[1,1]:
                result = morph.binary_shrink(a)
                self.assertEqual(np.sum(result),1)
    
    def test_08_labels(self):
        '''Run a labels matrix through shrink with two touching objects'''
        labels = np.zeros((10,10),int)
        labels[2:8,2:5] = 1
        labels[2:8,5:8] = 2
        result = morph.binary_shrink(labels)
        self.assertFalse(np.any(result[labels==0] > 0))
        my_sum = fix(scind.sum(result>0, labels, np.arange(1,3,dtype=np.int32)))
        self.assertTrue(np.all(my_sum == 1))
        
class TestCpmaximum(unittest.TestCase):
    def test_01_zeros(self):
        input = np.zeros((10,10))
        output = morph.cpmaximum(input)
        self.assertTrue(np.all(output==input))
    
    def test_01_ones(self):
        input = np.ones((10,10))
        output = morph.cpmaximum(input)
        self.assertTrue(np.all(np.abs(output-input)<=np.finfo(float).eps))

    def test_02_center_point(self):
        input = np.zeros((9,9))
        input[4,4] = 1
        expected = np.zeros((9,9))
        expected[3:6,3:6] = 1
        structure = np.ones((3,3),dtype=bool)
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(np.all(output==expected))
    
    def test_03_corner_point(self):
        input = np.zeros((9,9))
        input[0,0]=1
        expected = np.zeros((9,9))
        expected[:2,:2]=1
        structure = np.ones((3,3),dtype=bool)
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(np.all(output==expected))

    def test_04_structure(self):
        input = np.zeros((9,9))
        input[0,0]=1
        input[4,4]=1
        structure = np.zeros((3,3),dtype=bool)
        structure[0,0]=1
        expected = np.zeros((9,9))
        expected[1,1]=1
        expected[5,5]=1
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(np.all(output[1:,1:]==expected[1:,1:]))

    def test_05_big_structure(self):
        big_disk = morph.strel_disk(10).astype(bool)
        input = np.zeros((1001,1001))
        input[500,500] = 1
        expected = np.zeros((1001,1001))
        expected[490:551,490:551][big_disk]=1
        output = morph.cpmaximum(input,big_disk)
        self.assertTrue(np.all(output == expected))

class TestRelabel(unittest.TestCase):
    def test_00_relabel_zeros(self):
        input = np.zeros((10,10),int)
        output,count = morph.relabel(input)
        self.assertTrue(np.all(input==output))
        self.assertEqual(count, 0)
    
    def test_01_relabel_one(self):
        input = np.zeros((10,10),int)
        input[3:6,3:6]=1
        output,count = morph.relabel(input)
        self.assertTrue(np.all(input==output))
        self.assertEqual(count,1)
    
    def test_02_relabel_two_to_one(self):
        input = np.zeros((10,10),int)
        input[3:6,3:6]=2
        output,count = morph.relabel(input)
        self.assertTrue(np.all((output==1)[input==2]))
        self.assertTrue(np.all((input==output)[input!=2]))
        self.assertEqual(count,1)
    
    def test_03_relabel_gap(self):
        input = np.zeros((20,20),int)
        input[3:6,3:6]=1
        input[3:6,12:15]=3
        output,count = morph.relabel(input)
        self.assertTrue(np.all((output==2)[input==3]))
        self.assertTrue(np.all((input==output)[input!=3]))
        self.assertEqual(count,2)

class TestConvexHull(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure convex_hull can handle an empty array"""
        result,counts = morph.convex_hull(np.zeros((10,10),int), [])
        self.assertEqual(np.product(result.shape),0)
        self.assertEqual(np.product(counts.shape),0)
    
    def test_01_01_zeros(self):
        """Make sure convex_hull can work if a label has no points"""
        result,counts = morph.convex_hull(np.zeros((10,10),int), [1])
        self.assertEqual(np.product(result.shape),0)
        self.assertEqual(np.product(counts.shape),1)
        self.assertEqual(counts[0],0)
    
    def test_01_02_point(self):
        """Make sure convex_hull can handle the degenerate case of one point"""
        labels = np.zeros((10,10),int)
        labels[4,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(result.shape,(1,3))
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],4)
        self.assertEqual(result[0,2],5)
        self.assertEqual(counts[0],1)
    
    def test_01_030_line(self):
        """Make sure convex_hull can handle the degenerate case of a line"""
        labels = np.zeros((10,10),int)
        labels[2:8,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],2)
        self.assertEqual(result.shape,(2,3))
        self.assertTrue(np.all(result[:,0]==1))
        self.assertTrue(result[0,1] in (2,7))
        self.assertTrue(result[1,1] in (2,7))
        self.assertTrue(np.all(result[:,2]==5))
    
    def test_01_031_odd_line(self):
        """Make sure convex_hull can handle the degenerate case of a line with odd length
        
        This is a regression test: the line has a point in the center if
        it's odd and the sign of the difference of that point is zero
        which causes it to be included in the hull.
        """
        labels = np.zeros((10,10),int)
        labels[2:7,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],2)
        self.assertEqual(result.shape,(2,3))
        self.assertTrue(np.all(result[:,0]==1))
        self.assertTrue(result[0,1] in (2,6))
        self.assertTrue(result[1,1] in (2,6))
        self.assertTrue(np.all(result[:,2]==5))
    
    def test_01_04_square(self):
        """Make sure convex_hull can handle a square which is not degenerate"""
        labels = np.zeros((10,10),int)
        labels[2:7,3:8] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],4)
        order = np.lexsort((result[:,2], result[:,1]))
        result = result[order,:]
        expected = np.array([[1,2,3],
                                [1,2,7],
                                [1,6,3],
                                [1,6,7]])
        self.assertTrue(np.all(result==expected))
    
    def test_02_01_out_of_order(self):
        """Make sure convex_hull can handle out of order indices"""
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[5,6] = 2
        result,counts = morph.convex_hull(labels,[2,1])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(np.all(counts==1))
        
        expected = np.array([[2,5,6],[1,2,3]])
        self.assertTrue(np.all(result == expected))
    
    def test_02_02_out_of_order(self):
        """Make sure convex_hull can handle out of order indices
        that require different #s of loop iterations"""
        
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[1:7,4:8] = 2
        result,counts = morph.convex_hull(labels, [2,1])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(np.all(counts==(4,1)))
        self.assertEqual(result.shape,(5,3))
        order = np.lexsort((result[:,2],result[:,1],
                               np.array([0,2,1])[result[:,0]]))
        result = result[order,:]
        expected = np.array([[2,1,4],
                                [2,1,7],
                                [2,6,4],
                                [2,6,7],
                                [1,2,3]])
        self.assertTrue(np.all(result==expected))
    
    def test_02_03_two_squares(self):
        """Make sure convex_hull can handle two complex shapes"""
        labels = np.zeros((10,10),int)
        labels[1:5,3:7] = 1
        labels[6:10,1:7] = 2
        result,counts = morph.convex_hull(labels, [1,2])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(np.all(counts==(4,4)))
        order = np.lexsort((result[:,2],result[:,1],result[:,0]))
        result = result[order,:]
        expected = np.array([[1,1,3],[1,1,6],[1,4,3],[1,4,6],
                                [2,6,1],[2,6,6],[2,9,1],[2,9,6]])
        self.assertTrue(np.all(result==expected))
        
    def test_03_01_concave(self):
        """Make sure convex_hull handles a square with a concavity"""
        labels = np.zeros((10,10),int)
        labels[2:8,3:9] = 1
        labels[3:7,3] = 0
        labels[2:6,4] = 0
        labels[4:5,5] = 0
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],4)
        order = np.lexsort((result[:,2],result[:,1],result[:,0]))
        result = result[order,:]
        expected = np.array([[1,2,3],
                                [1,2,8],
                                [1,7,3],
                                [1,7,8]])
        self.assertTrue(np.all(result==expected))
        
    def test_04_01_regression(self):
        """The set of points given in this case yielded one in the interior"""
        np.random.seed(0)
        s = 10 # divide each image into this many mini-squares with a shape in each
        side = 250
        mini_side = side / s
        ct = 20
        labels = np.zeros((side,side),int)
        pts = np.zeros((s*s*ct,2),int)
        index = np.array(range(pts.shape[0])).astype(float)/float(ct)
        index = index.astype(int)
        idx = 0
        for i in range(0,side,mini_side):
            for j in range(0,side,mini_side):
                idx = idx+1
                # get ct+1 unique points
                p = np.random.uniform(low=0,high=mini_side,
                                         size=(ct+1,2)).astype(int)
                while True:
                    pu = np.unique(p[:,0]+p[:,1]*mini_side)
                    if pu.shape[0] == ct+1:
                        break
                    p[:pu.shape[0],0] = np.mod(pu,mini_side).astype(int)
                    p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                    p_size = (ct+1-pu.shape[0],2)
                    p[pu.shape[0],:] = np.random.uniform(low=0,
                                                            high=mini_side,
                                                            size=p_size)
                # Use the last point as the "center" and order
                # all of the other points according to their angles
                # to this "center"
                center = p[ct,:]
                v = p[:ct,:]-center
                angle = np.arctan2(v[:,0],v[:,1])
                order = np.lexsort((angle,))
                p = p[:ct][order]
                p[:,0] = p[:,0]+i
                p[:,1] = p[:,1]+j
                pts[(idx-1)*ct:idx*ct,:]=p
                #
                # draw lines on the labels
                #
                for k in range(ct):
                    morph.draw_line(labels, p[k,:], p[(k+1)%ct,:], idx)
        self.assertTrue(labels[5,106]==5)
        result,counts = morph.convex_hull(labels,np.array(range(100))+1)
        self.assertFalse(np.any(np.logical_and(result[:,1]==5,
                                                     result[:,2]==106)))
    
    def test_05_01_missing_labels(self):
        '''Ensure that there's an entry if a label has no corresponding points'''
        labels = np.zeros((10,10),int)
        labels[3:6,2:8] = 2
        result, counts = morph.convex_hull(labels, np.arange(2)+1)
        self.assertEqual(counts.shape[0], 2)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], 4)
        
    def test_06_01_regression_373(self):
        '''Regression test of IMG-374'''
        labels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        result, counts = morph.convex_hull(labels, np.array([1]))
        self.assertEqual(counts[0], 2)
        
    def test_06_02_same_point_twice(self):
        '''Regression test of convex_hull_ijv - same point twice in list'''
        
        ii = [79, 11, 65, 73, 42, 26, 46, 48, 14, 53, 73, 42, 59, 12, 59, 65,  7, 66, 84, 70]
        jj = [47, 97, 98,  0, 91, 49, 42, 85, 63, 19,  0,  9, 71, 15, 50, 98, 14, 46, 89, 47]
        h, c = morph.convex_hull_ijv(
            np.column_stack((ii, jj, np.ones(len(ii)))), [1])
        self.assertTrue(np.any((h[:,1] == 73) & (h[:,2] == 0)))

class TestConvexHullImage(unittest.TestCase):
    def test_00_00_zeros(self):
        image = np.zeros((10,13), bool)
        output = morph.convex_hull_image(image)
        self.assertTrue(np.all(output == False))
        
    def test_01_01_square(self):
        image = np.zeros((10,13), bool)
        image[2:5,3:8] = True
        output = morph.convex_hull_image(image)
        self.assertTrue(np.all(output == image))
    
    def test_01_02_concave(self):
        image = np.zeros((10,13), bool)
        image[2:5,3:8] = True
        image2 = image.copy()
        image2[4,4:7] = False
        output = morph.convex_hull_image(image2)
        self.assertTrue(np.all(output == image))
        
class TestMinimumEnclosingCircle(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure minimum_enclosing_circle can handle an empty array"""
        center,radius = morph.minimum_enclosing_circle(np.zeros((10,10),int), [])
        self.assertEqual(np.product(center.shape),0)
        self.assertEqual(np.product(radius.shape),0)
    
    def test_01_01_01_zeros(self):
        """Make sure minimum_enclosing_circle can work if a label has no points"""
        center,radius = morph.minimum_enclosing_circle(np.zeros((10,10),int), [1])
        self.assertEqual(center.shape,(1,2))
        self.assertEqual(np.product(radius.shape),1)
        self.assertEqual(radius[0],0)
    
    def test_01_01_02_zeros(self):
        """Make sure minimum_enclosing_circle can work if one of two labels has no points
        
        This is a regression test of a bug
        """
        labels = np.zeros((10,10), int)
        labels[2,2:5] = 3
        labels[2,6:9] = 4
        hull_and_point_count = morph.convex_hull(labels)
        center,radius = morph.minimum_enclosing_circle(
            labels,
            hull_and_point_count=hull_and_point_count)
        self.assertEqual(center.shape,(2,2))
        self.assertEqual(np.product(radius.shape),2)
    
    def test_01_02_point(self):
        """Make sure minimum_enclosing_circle can handle the degenerate case of one point"""
        labels = np.zeros((10,10),int)
        labels[4,5] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertEqual(center.shape,(1,2))
        self.assertEqual(radius.shape,(1,))
        self.assertTrue(np.all(center==np.array([(4,5)])))
        self.assertEqual(radius[0],0)
    
    def test_01_03_line(self):
        """Make sure minimum_enclosing_circle can handle the degenerate case of a line"""
        labels = np.zeros((10,10),int)
        labels[2:7,5] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertTrue(np.all(center==np.array([(4,5)])))
        self.assertEqual(radius[0],2)
    
    def test_01_04_square(self):
        """Make sure minimum_enclosing_circle can handle a square which is not degenerate"""
        labels = np.zeros((10,10),int)
        labels[2:7,3:8] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertTrue(np.all(center==np.array([(4,5)])))
        self.assertAlmostEqual(radius[0],np.sqrt(8))
    
    def test_02_01_out_of_order(self):
        """Make sure minimum_enclosing_circle can handle out of order indices"""
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[5,6] = 2
        center,radius = morph.minimum_enclosing_circle(labels,[2,1])
        self.assertEqual(center.shape,(2,2))
        
        expected_center = np.array(((5,6),(2,3)))
        self.assertTrue(np.all(center == expected_center))
    
    def test_02_02_out_of_order(self):
        """Make sure minimum_enclosing_circle can handle out of order indices
        that require different #s of loop iterations"""
        
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[1:6,4:9] = 2
        center,result = morph.minimum_enclosing_circle(labels, [2,1])
        expected_center = np.array(((3,6),(2,3)))
        self.assertTrue(np.all(center == expected_center))
    
    def test_03_01_random_polygons(self):
        """Test minimum_enclosing_circle on 250 random dodecagons"""
        np.random.seed(0)
        s = 10 # divide each image into this many mini-squares with a shape in each
        side = 250
        mini_side = side / s
        ct = 20
        #
        # We keep going until we get at least 10 multi-edge cases -
        # polygons where the minimum enclosing circle intersects 3+ vertices
        #
        n_multi_edge = 0
        while n_multi_edge < 10:
            labels = np.zeros((side,side),int)
            pts = np.zeros((s*s*ct,2),int)
            index = np.array(range(pts.shape[0])).astype(float)/float(ct)
            index = index.astype(int)
            idx = 0
            for i in range(0,side,mini_side):
                for j in range(0,side,mini_side):
                    idx = idx+1
                    # get ct+1 unique points
                    p = np.random.uniform(low=0,high=mini_side,
                                             size=(ct+1,2)).astype(int)
                    while True:
                        pu = np.unique(p[:,0]+p[:,1]*mini_side)
                        if pu.shape[0] == ct+1:
                            break
                        p[:pu.shape[0],0] = np.mod(pu,mini_side).astype(int)
                        p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                        p_size = (ct+1-pu.shape[0],2)
                        p[pu.shape[0],:] = np.random.uniform(low=0,
                                                                high=mini_side,
                                                                size=p_size)
                    # Use the last point as the "center" and order
                    # all of the other points according to their angles
                    # to this "center"
                    center = p[ct,:]
                    v = p[:ct,:]-center
                    angle = np.arctan2(v[:,0],v[:,1])
                    order = np.lexsort((angle,))
                    p = p[:ct][order]
                    p[:,0] = p[:,0]+i
                    p[:,1] = p[:,1]+j
                    pts[(idx-1)*ct:idx*ct,:]=p
                    #
                    # draw lines on the labels
                    #
                    for k in range(ct):
                        morph.draw_line(labels, p[k,:], p[(k+1)%ct,:], idx)
            center,radius = morph.minimum_enclosing_circle(labels, 
                                                           np.array(range(s**2))+1)
            epsilon = .000001
            center_per_pt = center[index]
            radius_per_pt = radius[index]
            distance_from_center = np.sqrt(np.sum((pts.astype(float)-
                                                         center_per_pt)**2,1))
            #
            # All points must be within the enclosing circle
            #
            self.assertTrue(np.all(distance_from_center - epsilon < radius_per_pt))
            pt_on_edge = np.abs(distance_from_center - radius_per_pt)<epsilon
            count_pt_on_edge = scind.sum(pt_on_edge,
                                                 index,
                                                 np.array(range(s**2),dtype=np.int32))
            count_pt_on_edge = np.array(count_pt_on_edge)
            #
            # Every dodecagon must have at least 2 points on the edge.
            #
            self.assertTrue(np.all(count_pt_on_edge>=2))
            #
            # Count the multi_edge cases
            #
            n_multi_edge += np.sum(count_pt_on_edge>=3)

class TestEllipseFromSecondMoments(unittest.TestCase):
    def assertWithinFraction(self, actual, expected, 
                             fraction=.001, message=None):
        """Assert that a "correlation" of the actual value to the expected is within the fraction
        
        actual - the value as calculated
        expected - the expected value of the variable
        fraction - the fractional difference of the two
        message - message to print on failure
        
        We divide the absolute difference by 1/2 of the sum of the variables
        to get our measurement.
        """
        measurement = abs(actual-expected)/(2*(actual+expected))
        self.assertTrue(measurement < fraction,
                        "%(actual)f != %(expected)f by the measure, abs(%(actual)f-%(expected)f)) / 2(%(actual)f + %(expected)f)"%(locals()))
        
    def test_00_00_zeros(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.zeros((10,10)),
                                              np.zeros((10,10),int),
                                              [])
        self.assertEqual(centers.shape,(0,2))
        self.assertEqual(eccentricity.shape[0],0)
        self.assertEqual(major_axis_length.shape[0],0)
        self.assertEqual(minor_axis_length.shape[0],0)
    
    def test_00_01_zeros(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.zeros((10,10)),
                                              np.zeros((10,10),int),
                                              [1])
        self.assertEqual(centers.shape,(1,2))
        self.assertEqual(eccentricity.shape[0],1)
        self.assertEqual(major_axis_length.shape[0],1)
        self.assertEqual(minor_axis_length.shape[0],1)
    
    def test_01_01_rectangle(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones((10,20)),
                                              np.ones((10,20),int),
                                              [1])
        self.assertEqual(centers.shape,(1,2))
        self.assertEqual(eccentricity.shape[0],1)
        self.assertEqual(major_axis_length.shape[0],1)
        self.assertEqual(minor_axis_length.shape[0],1)
        self.assertAlmostEqual(eccentricity[0],.866,2)
        self.assertAlmostEqual(centers[0,0],4.5)
        self.assertAlmostEqual(centers[0,1],9.5)
        self.assertWithinFraction(major_axis_length[0],23.0940,.001)
        self.assertWithinFraction(minor_axis_length[0],11.5470,.001)
        self.assertAlmostEqual(theta[0],0)
    
    def test_01_02_circle(self):
        img = np.zeros((101,101),int)
        y,x = np.mgrid[-50:51,-50:51]
        img[x*x+y*y<=2500] = 1
        centers,eccentricity,major_axis_length, minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones((101,101)),img,[1])
        self.assertAlmostEqual(eccentricity[0],0)
        self.assertWithinFraction(major_axis_length[0],100,.001)
        self.assertWithinFraction(minor_axis_length[0],100,.001)
    
    def test_01_03_blob(self):
        '''Regression test a blob against Matlab measurements'''
        blob = np.array(
            [[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
             [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
             [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0]])
        centers,eccentricity,major_axis_length, minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones(blob.shape),blob,[1])
        self.assertAlmostEqual(major_axis_length[0],37.55,1)
        self.assertAlmostEqual(minor_axis_length[0],18.99,1)
        self.assertAlmostEqual(eccentricity[0],0.8627,2)
        self.assertAlmostEqual(centers[0,1],14.1689,2)
        self.assertAlmostEqual(centers[0,0],14.8691,2)
        
    def test_02_01_compactness_square(self):
        image = np.zeros((9,9), int)
        image[1:8,1:8] = 1
        compactness = morph.ellipse_from_second_moments(
            np.ones(image.shape), image, [1], True)[-1]
        i,j = np.mgrid[0:9, 0:9]
        v_i = np.var(i[image > 0])
        v_j = np.var(j[image > 0])
        v = v_i + v_j
        area = np.sum(image > 0)
        expected = 2 * np.pi * v / area
        self.assertAlmostEqual(compactness, expected)
        

class TestCalculateExtents(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure calculate_extents doesn't throw an exception if no image"""
        extents = morph.calculate_extents(np.zeros((10,10),int), [1])
    
    def test_01_01_square(self):
        """A square should have an extent of 1"""
        labels = np.zeros((10,10),int)
        labels[1:8,2:9]=1
        extents = morph.calculate_extents(labels,[1])
        self.assertAlmostEqual(extents,1)
    
    def test_01_02_circle(self):
        """A circle should have an extent of pi/4"""
        labels = np.zeros((1001,1001),int)
        y,x = np.mgrid[-500:501,-500:501]
        labels[x*x+y*y<=250000] = 1
        extents = morph.calculate_extents(labels,[1])
        self.assertAlmostEqual(extents,np.pi/4,2)
        
    def test_01_03_two_objects(self):
        '''Make sure that calculate_extents works with more than one object
        
        Regression test of a bug: was computing area like this:
        scind.sum(labels, labels, indexes)
        which works for the object that's labeled "1", but is 2x for 2, 3x
        for 3, etc... oops.
        '''
        labels = np.zeros((10,20), int)
        labels[3:7, 2:5] = 1
        labels[3:5, 5:8] = 1
        labels[2:8, 13:17] = 2
        extents = morph.calculate_extents(labels, [1,2])
        self.assertEqual(len(extents), 2)
        self.assertAlmostEqual(extents[0], .75)
        self.assertAlmostEqual(extents[1], 1)
        
class TestMedianOfLabels(unittest.TestCase):
    def test_00_00_zeros(self):
        result = morph.median_of_labels(np.zeros((10,10)), 
                                        np.zeros((10,10), int),
                                        np.zeros(0, int))
        self.assertEqual(len(result), 0)
        
    def test_00_01_empty(self):
        result = morph.median_of_labels(np.zeros((10,10)), 
                                        np.zeros((10,10), int),
                                        [1])
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))
        
    def test_01_01_one_odd(self):
        r = np.random.RandomState()
        r.seed(11)
        fill = r.uniform(size=25)
        img = np.zeros((10,10))
        labels = np.zeros((10,10), int)
        labels[3:8,3:8] = 1
        img[labels > 0] = fill
        result = morph.median_of_labels(img, labels, [ 1 ])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], np.median(fill))
        
    def test_01_02_one_even(self):
        r = np.random.RandomState()
        r.seed(12)
        fill = r.uniform(size=20)
        img = np.zeros((10,10))
        labels = np.zeros((10,10), int)
        labels[3:8,3:7] = 1
        img[labels > 0] = fill
        result = morph.median_of_labels(img, labels, [ 1 ])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], np.median(fill))
        
    def test_01_03_two(self):
        r = np.random.RandomState()
        r.seed(12)
        img = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        labels[3:8,3:7] = 1
        labels[3:8,13:18] = 2
        for i, fill in enumerate([r.uniform(size=20), r.uniform(size=25)]):
            img[labels == i+1] = fill
        result = morph.median_of_labels(img, labels, [ 1,2 ])
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], np.median(img[labels==1]))
        self.assertAlmostEqual(result[1], np.median(img[labels==2]))
        
        
class TestCalculatePerimeters(unittest.TestCase):
    def test_00_00_zeros(self):
        """The perimeters of a zeros matrix should be all zero"""
        perimeters = morph.calculate_perimeters(np.zeros((10,10),int),[1])
        self.assertEqual(perimeters,0)
    
    def test_01_01_square(self):
        """The perimeter of a square should be the sum of the sides"""
        
        labels = np.zeros((10,10),int)
        labels[1:9,1:9] = 1
        perimeter = morph.calculate_perimeters(labels, [1])
        self.assertEqual(perimeter, 4*8)
        
    def test_01_02_circle(self):
        """The perimeter of a circle should be pi * diameter"""
        labels = np.zeros((101,101),int)
        y,x = np.mgrid[-50:51,-50:51]
        labels[x*x+y*y<=2500] = 1
        perimeter = morph.calculate_perimeters(labels, [1])
        epsilon = 20
        self.assertTrue(perimeter-np.pi*101<epsilon)
        
    def test_01_03_on_edge(self):
        """Check the perimeter of objects touching edges of matrix"""
        labels = np.zeros((10,20), int)
        labels[:4,:4] = 1 # 4x4 square = 16 pixel perimeter
        labels[-4:,-2:] = 2 # 4x2 square = 2+2+4+4 = 12
        expected = [ 16, 12]
        perimeter = morph.calculate_perimeters(labels, [1,2])
        self.assertEqual(len(perimeter), 2)
        self.assertEqual(perimeter[0], expected[0])
        self.assertEqual(perimeter[1], expected[1])

class TestCalculateConvexArea(unittest.TestCase):
    def test_00_00_degenerate_zero(self):
        """The convex area of an empty labels matrix should be zero"""
        labels = np.zeros((10,10),int)
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],0)
    
    def test_00_01_degenerate_point(self):
        """The convex area of a point should be 1"""
        labels = np.zeros((10,10),int)
        labels[4,4] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],1)

    def test_00_02_degenerate_line(self):
        """The convex area of a line should be its length"""
        labels = np.zeros((10,10),int)
        labels[1:9,4] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],8)
    
    def test_01_01_square(self):
        """The convex area of a square should be its area"""
        labels = np.zeros((10,10),int)
        labels[1:9,1:9] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertAlmostEqual(result[0],64)
    
    def test_01_02_cross(self):
        """The convex area of a cross should be the area of the enclosing diamond
        
        The area of a diamond is 1/2 of the area of the enclosing bounding box
        """
        labels = np.zeros((10,10),int)
        labels[1:9,4] = 1
        labels[4,1:9] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertAlmostEqual(result[0],32)
    
    def test_02_01_degenerate_point_and_line(self):
        """Test a degenerate point and line in the same image, out of order"""
        labels = np.zeros((10,10),int)
        labels[1,1] = 1
        labels[1:9,4] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertEqual(result[0],8)
        self.assertEqual(result[1],1)
    
    def test_02_02_degenerate_point_and_square(self):
        """Test a degenerate point and a square in the same image"""
        labels = np.zeros((10,10),int)
        labels[1,1] = 1
        labels[3:8,4:9] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertEqual(result[1],1)
        self.assertAlmostEqual(result[0],25)
    
    def test_02_03_square_and_cross(self):
        """Test two non-degenerate figures"""
        labels = np.zeros((20,10),int)
        labels[1:9,1:9] = 1
        labels[11:19,4] = 2
        labels[14,1:9] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertAlmostEqual(result[0],32)
        self.assertAlmostEqual(result[1],64)

class TestEulerNumber(unittest.TestCase):
    def test_00_00_even_zeros(self):
        labels = np.zeros((10,12),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_00_01_odd_zeros(self):
        labels = np.zeros((11,13),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_01_00_square(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],1)
        
    def test_01_01_square_with_hole(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        labels[3:6,3:6] = 0
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_01_02_square_with_two_holes(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        labels[2:4,2:8] = 0
        labels[6:8,2:8] = 0
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],-1)
    
    def test_02_01_square_touches_border(self):
        labels = np.ones((10,10),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],1)
    
    def test_03_01_two_objects(self):
        labels = np.zeros((10,10), int)
        # First object has a hole - Euler # is zero
        labels[1:4,1:4] = 1
        labels[2,2] = 0
        # Second object has no hole - Euler # is 1
        labels[5:8,5:8] = 2
        result = morph.euler_number(labels, [1,2])
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)

class TestWhiteTophat(unittest.TestCase):
    '''Test the white_tophat function'''
    def test_01_01_zeros(self):
        '''Test white_tophat on an image of all zeros'''
        result = morph.white_tophat(np.zeros((10,10)), 1)
        self.assertTrue(np.all(result==0))
    
    def test_01_02_ones(self):
        '''Test white_tophat on an image of all ones'''
        result = morph.white_tophat(np.ones((10,10)), 1)
        self.assertTrue(np.all(result==0))
    
    def test_01_03_edge(self):
        '''Test white_tophat on an image whose edge is zeros.
        
        The image should erode off the sides to a depth of 1
        and then should dilate, leaving the four corners as one
        '''
        image = np.zeros((10,10))
        image[1:9,1:9] = 1
        expected = np.zeros((10,10))
        expected[1,1] = 1
        expected[8,8] = 1
        expected[1,8] = 1
        expected[8,1] = 1
        result = morph.white_tophat(image, 1)
        self.assertTrue(np.all(result==expected))
    
    def test_01_04_random(self):
        '''Test that a random image has the same value as Matlab'''
        data = base64.b64decode(
            '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'
            '3XFcs7s/VgfGl78/1D/qOMjU2VXkP63mN5Mgle0/aA5hM/l+uD8EyLTvMFH'
            'tP9h31Sdu9bI/fSC4K8Ej5j+bZXP/Yy/nP5TRCc3jMcQ/wKIrWdbYoT+pfU'
            'Z3cX7mP1n5Fmd37us/WD18Z3833D8njar5DUzsP2ymbX/W5uc/djqRyDYl0'
            'z/qWvahyvrTP62fVz6zDOI/azSv9Qkc7T+YT9+enubDP/DrGHx3tqQ/UpFo'
            'm6G92z9Q1TnCXqDJP7zssnGbhN8/VS3uvkgM5j+Pa7fVeLTiP4IU97fHaN0'
            '/qXQZhZf74z+qktjZt+brP5DddNz1Dak/6LnnNKrY5j9VwV+UNtfsP0aVtu'
            'A6WNQ/30NGCX925D/RDau7ZPvvPy8SyRHWdeY/QNxHWLWRqz+CZKZA/SjvP'
            'yn4v1P92uA/3F8Q2HCP7z8kIQRry4viPxNqebuC0Oo/euy6GmB66D8GH4YS'
            'VVTgP+xuPRhNN+k/cRL7YN7+6z8zWZbhs3fqPxaMPabnEtA/Wcrye9Za5j/'
            'lMIaUW0DkPwqEGMMFQeU/wrR9+Uxy6T+iCVQ5rBvZPyYihQuyC+U/5nYnry'
            'bx0z8wY5jciPzhP65FiU6u1eQ/urfL8Cd22T+bspgPlb7tPxy38b5TF+Q/h'
            'HXUGkcy4z8OGAhjSEDhP+DaMtUZb+c/7yZx/6HW5j8cgTqj0tzEPzwgNV7W'
            'UsM/X6LwkDvS7D+sUlG39R/eP9hiY7/D9Lk/ZZu3ezL94j990CDicTvoP0A'
            '8AyZ3a4U/sKizUYL7vz8DaAIqF4bkP7iiy4PXtbE/vK/uX5FFwT/AFqyvyz'
            'icP+QgVszaNNY/oDhF1BYCuz+wmsMnDgzjP+mUWuaH1u4/qMaZmADOtz+6p'
            'G179lTTPwpMDk7G4Ow/1uiN0ob94T/9JxsTWtnlP+PqQvFHYuM/gm8o0OOR'
            '3T9toTlcJ+XgP4xYm73q/9g/RhOCsXDd2D8IRS92/xLvPzolVKKy7Oo//l6'
            '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'
            'p9kug/bbQGWCh25T+6xHr/JXfcP+rqgX9OYtg/ihr1zrPL7T9JyoXyXJXuP'
            'wIKx16+9tk/gvlhIQrk6D9EN4EtPFjOPzhGB0YNkeQ/d5iMSyq07D+M9GBT'
            'FAvoP8DMc84h6Iw/usp9Dxs62j9oRiMtIMDNPxarycPPw9I/gElN19ezjz8'
            'eYc+VolbZP6uzF/OjaeE/spbc72LE0z/08gidfcfWP8i5xhFeWus/XmZIw/'
            'oa5D/qH43w2XTpPxKSL6rs9tU/XDKVePWn5D+DJLlYc3juPyKYSm3apdQ/C'
            'tOI2/Xw1j8rnjI8Os3hPzH9zTKmHOQ/CIj2pOql2z9Uv7cwkjrAPwIuXZZB'
            'SNo/sM3K34xRwD/gXhukw4/pP0D/S9dyJrU/0J+3+jjtyz9eopXJrUHZP6C'
            'SuXQfoLU/kIN1+F0uvD8AsUI+CcTTP+hoHjXdX9g/CW3/+euO6z/uGraBnp'
            'XQP6g9nJh3tMo/YWvuUdyN6j9QR/tM7nXIPz73ZE72WOk/WG1lrOV+xj94c'
            'rE6pRC1Py6R7Mms1uk/lBEI/aI5zT8p0JE1uVfrP1Nvsp6syec/LZ5leCYI'
            '6T+g+OKGp3SkP0gAieuJaN0/gL8OEzwgxz8eQWS7ZebcP5DXZRYVCt8/DAL'
            'nAbyY2z+iBcO0DM/bPwTNnp1/4ME/sP/P0Fm7wD9EjYbNRyntP9hFZFKEYb'
            'Q/IChMYXbdoj+UbQ+TnfHPP8JYoAnjtNI/bOh6nLeizD/pWROHvSPhP4QxX'
            '1Fbs8c/7t3EzvYt6j/JsgjIuuHuPxzHOx6BWeI/JlA3reSH2D8wzARsIKe3'
            'P8DZ7fR+ddw/uPbXh74avz9FhDkPTB7mP7WEFi1qDuw/Bqtbplsw6D8wOBy'
            'jQnqgP/amk9W3IeE/K4Qdo1cq6D+4M+v8SxPHPzCE7skKPaU/5B3X6qt6zT'
            '+2NoROH5DoP0Ns+Owt5eM/nIQUHsGU5D/cQDzHwqXMP6hY8x1azbA/Gnu8V'
            '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'
            'T0No/mzkBETEK6j8=')
        expected = base64.b64decode(
            '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'
            'HxNU/TugfQWuE5z8=')
        data = np.fromstring(data)
        data.shape = (40,40)
        expected = np.fromstring(expected)
        expected.shape = (40,40)
        result = morph.white_tophat(data, 3)
        #
        # Matlab will give different results because of edge stuff
        #
        self.assertTrue(np.all(result[8:-8,8:-8] == expected[8:-8,8:-8]))
        #
        # Calculate 0:0 by hand
        #
        s = morph.strel_disk(3) > 0
        corner = np.ones((4,4))
        for i in range(4):
            for j in range(4):
                for k in range(-3,4):
                    for l in range(-3,4):
                        if s[k+3,l+3] and i+k > 0 and j+l > 0:
                            corner[i,j] = min(corner[i,j],data[i+k,j+l])
        my_max = np.max(corner[s[3:,3:]])
        my_value = data[0,0] - my_max
        self.assertEqual(my_value, result[0,0])
    
    def test_02_01_mask(self):
        '''Test white_tophat, masking the pixels that would erode'''
        image = np.zeros((10,10))
        image[1:9,1:9] = 1
        mask = image != 0
        result = morph.white_tophat(image, 1, mask)
        self.assertTrue(np.all(result==0))
    
class TestRegionalMaximum(unittest.TestCase):
    def test_00_00_zeros(self):
        '''An array of all zeros has a regional maximum of its center'''
        result = morph.regional_maximum(np.zeros((11,11)))
        self.assertEqual(np.sum(result),1)
        self.assertTrue(result[5,5])
    
    def test_00_01_zeros_with_mask(self):
        result = morph.regional_maximum(np.zeros((10,10)),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_single_maximum(self):
        '''Test that the regional maximum of a gradient from 5,5 is 5,5'''
        #
        # Create a gradient of distance from the point 5,5 in an 11x11 array
        #
        i,j = np.mgrid[-5:6,-5:6].astype(float) / 5
        image = 1 - i**2 - j**2
        result = morph.regional_maximum(image)
        self.assertTrue(result[5,5])
        self.assertTrue(np.all(result[image != np.max(image)]==False))
    
    def test_01_02_two_maxima(self):
        '''Test an image with two maxima'''
        i,j = np.mgrid[-5:6,-5:6].astype(float) / 5
        half_image = 1 - i**2 - j**2
        image = np.zeros((11,22))
        image[:,:11]=half_image
        image[:,11:]=half_image
        result = morph.regional_maximum(image)
        self.assertTrue(result[5,5])
        self.assertTrue(result[5,-6])
        self.assertTrue(np.all(result[image != np.max(image)]==False))
    
    def test_02_01_mask(self):
        '''Test that a mask eliminates one of the maxima'''
        i,j = np.mgrid[-5:6,-5:6].astype(float) / 5
        half_image = 1 - i**2 - j**2
        image = np.zeros((11,22))
        image[:,:11]=half_image
        image[:,11:]=half_image
        mask = np.ones(image.shape, bool)
        mask[4,5] = False
        result = morph.regional_maximum(image,mask)
        self.assertFalse(result[5,5])
        self.assertTrue(result[5,-6])
        self.assertTrue(np.all(result[image != np.max(image)]==False))

class TestBlackTophat(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test black tophat on an array of all zeros'''
        result = morph.black_tophat(np.zeros((10,10)), 1)
        self.assertTrue(np.all(result==0))
        
    def test_00_01_zeros_masked(self):
        '''Test black tophat on an array that is completely masked'''
        result = morph.black_tophat(np.zeros((10,10)),1,np.zeros((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_01_01_single(self):
        '''Test black tophat of a single minimum'''
        result = morph.black_tophat(np.array([[.9,.8,.7],
                                              [.9,.5,.7],
                                              [.7,.8,.8]]),1)
        #
        # The edges should not be affected by the border
        #
        expected = np.array([[0,0,.1],[0,.3,.1],[.1,0,0]])
        self.assertTrue(np.all(np.abs(result - expected)<.00000001))
    
    def test_02_01_mask(self):
        '''Test black tophat with a mask'''
        image = np.array([[.9, .8, .7],[.9,.5,.7],[.7,.8,.8]])
        mask = np.array([[1,1,0],[1,1,0],[1,0,1]],bool)
        expected = np.array([[0,.1,0],[0,.4,0],[.2,0,0]])
        result = morph.black_tophat(image, 1, mask)
        self.assertTrue(np.all(np.abs(result[mask]-expected[mask])<.0000001))

class TestClosing(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test closing on an array of all zeros'''
        result = morph.closing(np.zeros((10,10)), 1)
        self.assertTrue(np.all(result==0))
        
    def test_00_01_zeros_masked(self):
        '''Test closing on an array that is completely masked'''
        result = morph.closing(np.zeros((10,10)),1,np.zeros((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_01_01_single(self):
        '''Test closing of a single minimum'''
        result = morph.closing(np.array([[.9,.8,.7],
                                         [.9,.5,.7],
                                         [.7,.8,.8]]),1)
        #
        # The edges should not be affected by the border
        #
        expected = np.array([[.9,.8,.8],[.9,.8,.8],[.8,.8,.8]])
        self.assertTrue(np.all(np.abs(result - expected)<.00000001))
    
    def test_02_01_mask(self):
        '''Test closing with a mask'''
        image = np.array([[.9, .8, .7],[.9,.5,.7],[.7,.8,.8]])
        mask = np.array([[1,1,0],[1,1,0],[1,0,1]],bool)
        expected = np.array([[.9,.9,.7],[.9,.9,.7],[.9,.8,.8]])
        result = morph.closing(image, 1, mask)
        self.assertTrue(np.all(np.abs(result[mask]-expected[mask])<.0000001))
        
    def test_03_01_8_connected(self):
        '''Test closing with an 8-connected structuring element'''
        result = morph.closing(np.array([[.9,.8,.7],
                                         [.9,.5,.7],
                                         [.7,.8,.8]]))
        expected = np.array([[.9,.8,.8],[.9,.8,.8],[.9,.8,.8]])
        self.assertTrue(np.all(np.abs(result - expected)<.00000001))

class TestBranchpoints(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test branchpoints on an array of all zeros'''
        result = morph.branchpoints(np.zeros((9,11), bool))
        self.assertTrue(np.all(result == False))
        
    def test_00_01_zeros_masked(self):
        '''Test branchpoints on an array that is completely masked'''
        result = morph.branchpoints(np.zeros((10,10),bool),
                                    np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_branchpoints_positive(self):
        '''Test branchpoints on positive cases'''
        image = np.array([[1,0,0,1,0,1,0,1,0,1,0,0,1],
                          [0,1,0,1,0,0,1,0,1,0,1,1,0],
                          [1,0,1,0,1,1,0,1,1,1,0,0,1]],bool)
        result = morph.branchpoints(image)
        self.assertTrue(np.all(image[1,:] == result[1,:]))
    
    def test_01_02_branchpoints_negative(self):
        '''Test branchpoints on negative cases'''
        image = np.array([[1,0,0,0,1,0,0,0,1,0,1,0,1],
                          [0,1,0,0,1,0,1,1,1,0,0,1,0],
                          [0,0,1,0,1,0,0,0,0,0,0,0,0]],bool)
        result = morph.branchpoints(image)
        self.assertTrue(np.all(result==False))
        
    def test_02_01_branchpoints_masked(self):
        '''Test that masking defeats branchpoints'''
        image = np.array([[1,0,0,1,0,1,0,1,1,1,0,0,1],
                          [0,1,0,1,0,0,1,0,1,0,1,1,0],
                          [1,0,1,1,0,1,0,1,1,1,0,0,1]],bool)
        mask  = np.array([[0,1,1,1,1,1,1,0,0,0,1,1,0],
                          [1,1,1,1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,0,1,0,1,1,0,0,1,1,1]],bool)
        result = morph.branchpoints(image, mask)
        self.assertTrue(np.all(result[mask]==False))
        
class TestBridge(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test bridge on an array of all zeros'''
        result = morph.bridge(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test bridge on an array that is completely masked'''
        result = morph.bridge(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_bridge_positive(self):
        '''Test some typical positive cases of bridging'''
        image = np.array([[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1]],bool)
        expected = np.array([[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0],
                             [0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1],
                             [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1]],bool)
        result = morph.bridge(image)
        self.assertTrue(np.all(result==expected))
    
    def test_01_02_bridge_negative(self):
        '''Test some typical negative cases of bridging'''
        image = np.array([[1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0],
                          [0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0],
                          [0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1]],bool)

        expected = np.array([[1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,0],
                             [0,0,1,0,0,1,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1],
                             [0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1]],bool)
        result = morph.bridge(image)
        self.assertTrue(np.all(result==expected))

    def test_02_01_bridge_mask(self):
        '''Test that a masked pixel does not cause a bridge'''
        image = np.array([[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1]],bool)
        mask = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                         [1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],bool)
        expected = np.array([[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0],
                             [0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1],
                             [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1]],bool)
        result = morph.bridge(image,mask)
        self.assertTrue(np.all(result[mask]==expected[mask]))

class TestClean(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test clean on an array of all zeros'''
        result = morph.clean(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test clean on an array that is completely masked'''
        result = morph.clean(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_clean_positive(self):
        '''Test removal of a pixel using clean'''
        image = np.array([[0,0,0],[0,1,0],[0,0,0]],bool)
        self.assertTrue(np.all(morph.clean(image) == False))
    
    def test_01_02_clean_negative(self):
        '''Test patterns that should not clean'''
        image = np.array([[1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0],
                          [0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1]],bool)
        self.assertTrue(np.all(image == morph.clean(image)))
    
    def test_02_01_clean_edge(self):
        '''Test that clean removes isolated pixels on the edge of an image'''
        
        image = np.array([[1,0,1,0,1],
                          [0,0,0,0,0],
                          [1,0,0,0,1],
                          [0,0,0,0,0],
                          [1,0,1,0,1]],bool)
        self.assertTrue(np.all(morph.clean(image) == False))
        
    def test_02_02_clean_mask(self):
        '''Test that clean removes pixels adjoining a mask'''
        image = np.array([[0,0,0],[1,1,0],[0,0,0]],bool)
        mask  = np.array([[1,1,1],[0,1,1],[1,1,1]],bool)
        result= morph.clean(image,mask)
        self.assertEqual(result[1,1], False)
    
    def test_03_01_clean_labels(self):
        '''Test clean on a labels matrix where two single-pixel objects touch'''
        
        image = np.zeros((10,10), int)
        image[2,2] = 1
        image[2,3] = 2
        image[5:8,5:8] = 3
        result = morph.clean(image)
        self.assertTrue(np.all(result[image != 3] == 0))
        self.assertTrue(np.all(result[image==3] == 3))

class TestDiag(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test diag on an array of all zeros'''
        result = morph.diag(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test diag on an array that is completely masked'''
        result = morph.diag(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_diag_positive(self):
        '''Test all cases of diag filling in a pixel'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,1,0,0,0,0,1,0,0],
                          [0,1,0,0,0,0,1,0,0,1,0,1,0],
                          [0,0,0,0,0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,0,0,1,1,0,0,1,1,1,0],
                             [0,1,1,0,0,1,1,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.diag(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_diag_negative(self):
        '''Test patterns that should not diag'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,0,0,0,1,0,0,1,1,0,1,0,1,0],
                          [0,0,1,0,1,1,0,1,1,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        self.assertTrue(np.all(image == morph.diag(image)))
    
    def test_02_01_diag_edge(self):
        '''Test that diag works on edges'''
        
        image = np.array([[1,0,0,0,1],
                          [0,1,0,1,0],
                          [0,0,0,0,0],
                          [0,1,0,1,0],
                          [1,0,0,0,1]],bool)
        expected = np.array([[1,1,0,1,1],
                             [1,1,0,1,1],
                             [0,0,0,0,0],
                             [1,1,0,1,1],
                             [1,1,0,1,1]],bool)
        self.assertTrue(np.all(morph.diag(image) == expected))
        image = np.array([[0,1,0,1,0],
                          [1,0,0,0,1],
                          [0,0,0,0,0],
                          [1,0,0,0,1],
                          [0,1,0,1,0]],bool)
        self.assertTrue(np.all(morph.diag(image) == expected))
        
        
    def test_02_02_diag_mask(self):
        '''Test that diag connects if one of the pixels is masked'''
        image = np.array([[0,0,0],
                          [1,0,0],
                          [1,1,0]],bool)
        mask  = np.array([[1,1,1],
                          [1,1,1],
                          [0,1,1]],bool)
        result= morph.diag(image,mask)
        self.assertEqual(result[1,1], True)
        
class TestEndpoints(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test endpoints on an array of all zeros'''
        result = morph.endpoints(np.zeros((9,11), bool))
        self.assertTrue(np.all(result == False))
        
    def test_00_01_zeros_masked(self):
        '''Test endpoints on an array that is completely masked'''
        result = morph.endpoints(np.zeros((10,10),bool),
                                 np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_positive(self):
        '''Test positive endpoint cases'''
        image = np.array([[0,0,0,1,0,1,0,0,0,0,0],
                          [0,1,0,1,0,0,1,0,1,0,1],
                          [1,0,0,0,0,0,0,0,0,1,0]],bool)
        result = morph.endpoints(image)
        self.assertTrue(np.all(image[1,:] == result[1,:]))
    
    def test_01_02_negative(self):
        '''Test negative endpoint cases'''
        image = np.array([[0,0,1,0,0,1,0,0,0,0,0,1],
                          [0,1,0,1,0,1,0,0,1,1,0,1],
                          [1,0,0,0,1,0,0,1,0,0,1,0]],bool)
        result = morph.endpoints(image)
        self.assertTrue(np.all(result[1,:] == False))
        
    def test_02_02_mask(self):
        '''Test that masked positive pixels don't change the endpoint determination'''
        image = np.array([[0,0,1,1,0,1,0,1,0,1,0],
                          [0,1,0,1,0,0,1,0,1,0,1],
                          [1,0,0,0,1,0,0,0,0,1,0]],bool)
        mask  = np.array([[1,1,0,1,1,1,1,0,1,0,1],
                          [1,1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,1,0,1,1,1,1,1,1]],bool)
        result = morph.endpoints(image, mask)
        self.assertTrue(np.all(image[1,:] == result[1,:]))
    
class TestFill(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test fill on an array of all zeros'''
        result = morph.fill(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test fill on an array that is completely masked'''
        result = morph.fill(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_fill_positive(self):
        '''Test addition of a pixel using fill'''
        image = np.array([[1,1,1],[1,0,1],[1,1,1]],bool)
        self.assertTrue(np.all(morph.fill(image)))
    
    def test_01_02_fill_negative(self):
        '''Test patterns that should not fill'''
        image = np.array([[0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,0,0,1],
                          [1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1],
                          [1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0]],bool)
        self.assertTrue(np.all(image == morph.fill(image)))
    
    def test_02_01_fill_edge(self):
        '''Test that fill fills isolated pixels on an edge'''
        
        image = np.array([[0,1,0,1,0],
                          [1,1,1,1,1],
                          [0,1,1,1,0],
                          [1,1,1,1,1],
                          [0,1,0,1,0]],bool)
        self.assertTrue(np.all(morph.fill(image) == True))
        
    def test_02_02_fill_mask(self):
        '''Test that fill adds pixels if a neighbor is masked'''
        image = np.array([[1,1,1],
                          [0,0,1],
                          [1,1,1]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.fill(image,mask)
        self.assertEqual(result[1,1], True)

class TestHBreak(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test hbreak on an array of all zeros'''
        result = morph.hbreak(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test hbreak on an array that is completely masked'''
        result = morph.hbreak(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_hbreak_positive(self):
        '''Test break of a horizontal line'''
        image = np.array([[1,1,1],
                          [0,1,0],
                          [1,1,1]],bool)
        expected = np.array([[1,1,1],
                             [0,0,0],
                             [1,1,1]],bool)
        self.assertTrue(np.all(morph.hbreak(image)==expected))
    
    def test_01_02_hbreak_negative(self):
        '''Test patterns that should not hbreak'''
        image = np.array([[0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,0],
                          [0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0],
                          [0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,0]],bool)
        self.assertTrue(np.all(image == morph.hbreak(image)))
    
class TestVBreak(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test vbreak on an array of all zeros'''
        result = morph.vbreak(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test vbreak on an array that is completely masked'''
        result = morph.vbreak(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_vbreak_positive(self):
        '''Test break of a vertical line'''
        image = np.array([[1,0,1],
                          [1,1,1],
                          [1,0,1]],bool)
        expected = np.array([[1,0,1],
                             [1,0,1],
                             [1,0,1]],bool)
        self.assertTrue(np.all(morph.vbreak(image)==expected))
    
    def test_01_02_vbreak_negative(self):
        '''Test patterns that should not vbreak'''
        # stolen from hbreak
        image = np.array([[0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,0],
                          [0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0],
                          [0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,0]],bool)
        image = image.transpose()
        self.assertTrue(np.all(image == morph.vbreak(image)))
    
class TestMajority(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test majority on an array of all zeros'''
        result = morph.majority(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test majority on an array that is completely masked'''
        result = morph.majority(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_majority(self):
        '''Test majority on a random field'''
        np.random.seed(0)
        image = np.random.uniform(size=(10,10)) > .5
        expected = scipy.ndimage.convolve(image.astype(int), np.ones((3,3)), 
                                          mode='constant', cval=0) > 4.5
        result = morph.majority(image)
        self.assertTrue(np.all(result==expected))
                                        
class TestRemove(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test remove on an array of all zeros'''
        result = morph.remove(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test remove on an array that is completely masked'''
        result = morph.remove(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_remove_positive(self):
        '''Test removing a pixel'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,1,1,0,1,1,1,0],
                          [0,1,1,1,0,1,1,1,0,1,1,1,0],
                          [0,0,1,0,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,1,1,0,1,1,1,0],
                             [0,1,0,1,0,1,0,1,0,1,0,1,0],
                             [0,0,1,0,0,0,1,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.remove(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_remove_negative(self):
        '''Test patterns that should not diag'''
        image = np.array([[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,1,0,0,1,0,0,1,1,0,1,0,1,0],
                          [0,0,1,1,1,1,0,1,1,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]],bool)
        self.assertTrue(np.all(image == morph.remove(image)))
    
    def test_02_01_remove_edge(self):
        '''Test that remove does nothing'''
        
        image = np.array([[1,1,1,1,1],
                          [1,1,0,1,1],
                          [1,0,0,0,1],
                          [1,1,0,1,1],
                          [1,1,1,1,1]],bool)
        self.assertTrue(np.all(morph.remove(image) == image))
        
    def test_02_02_remove_mask(self):
        '''Test that a masked pixel does not cause a remove'''
        image = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.remove(image,mask)
        self.assertEqual(result[1,1], True)

class TestSkeleton(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test skeletonize on an array of all zeros'''
        result = morph.skeletonize(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test skeletonize on an array that is completely masked'''
        result = morph.skeletonize(np.zeros((10,10),bool),
                                   np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_rectangle(self):
        '''Test skeletonize on a rectangle'''
        image = np.zeros((9,15),bool)
        image[1:-1,1:-1] = True
        #
        # The result should be four diagonals from the
        # corners, meeting in a horizontal line
        #
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                             [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                             [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.skeletonize(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_hole(self):
        '''Test skeletonize on a rectangle with a hole in the middle'''
        image = np.zeros((9,15),bool)
        image[1:-1,1:-1] = True
        image[4,4:-4] = False
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.skeletonize(image)
        self.assertTrue(np.all(result == expected))
         
class TestSpur(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test spur on an array of all zeros'''
        result = morph.spur(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test spur on an array that is completely masked'''
        result = morph.spur(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_spur_positive(self):
        '''Test removing a spur pixel'''
        image    = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,1,0,0,0,1,0,1,0,0,0],
                             [0,1,1,1,0,1,0,0,1,0,0,0,1,0,0],
                             [0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,1,0,0,1,0,0,1,0,0,0,1,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_spur_negative(self):
        '''Test patterns that should not spur'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0],
                          [0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        l,count = scind.label(result,scind.generate_binary_structure(2, 2))
        self.assertEqual(count, 5)
        a = np.array(scind.sum(result,l,np.arange(4,dtype=np.int32)+1))
        self.assertTrue(np.all((a==1) | (a==4)))
    
    def test_02_01_spur_edge(self):
        '''Test that spurs on edges go away'''
        
        image = np.array([[1,0,0,1,0,0,1],
                          [0,1,0,1,0,1,0],
                          [0,0,1,1,1,0,0],
                          [1,1,1,1,1,1,1],
                          [0,0,1,1,1,0,0],
                          [0,1,0,1,0,1,0],
                          [1,0,0,1,0,0,1]],bool)
        expected = np.array([[0,0,0,0,0,0,0],
                             [0,1,0,1,0,1,0],
                             [0,0,1,1,1,0,0],
                             [0,1,1,1,1,1,0],
                             [0,0,1,1,1,0,0],
                             [0,1,0,1,0,1,0],
                             [0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        self.assertTrue(np.all(result == expected))
        
    def test_02_02_spur_mask(self):
        '''Test that a masked pixel does not prevent a spur remove'''
        image = np.array([[1,0,0],
                          [1,1,0],
                          [0,0,0]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.spur(image,mask)
        self.assertEqual(result[1,1], False)

class TestThicken(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test thicken on an array of all zeros'''
        result = morph.thicken(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test thicken on an array that is completely masked'''
        result = morph.thicken(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_thicken_positive(self):
        '''Test thickening positive cases'''
        image    = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,0,0,1,0,0,0],
                             [0,1,1,1,0,0,0,1,0,0,0,0,1,0,0],
                             [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,1,1,1,1,1,1,0,0],
                             [1,1,1,1,1,0,1,1,1,1,1,1,1,1,0],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1],
                             [0,0,0,0,0,1,1,1,0,0,0,0,1,1,1]],bool)
        result = morph.thicken(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_thicken_negative(self):
        '''Test patterns that should not thicken'''
        image = np.array([[1,1,0,1],
                          [0,0,0,0],
                          [1,1,1,1],
                          [0,0,0,0],
                          [1,1,0,1]],bool)
        result = morph.thicken(image)
        self.assertTrue(np.all(result==image))
    
    def test_02_01_thicken_edge(self):
        '''Test thickening to the edge'''
        
        image = np.zeros((5,5),bool)
        image[1:-1,1:-1] = True
        result = morph.thicken(image)
        self.assertTrue(np.all(result))
        
class TestThin(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test thin on an array of all zeros'''
        result = morph.thin(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test thin on an array that is completely masked'''
        result = morph.thin(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_bar(self):
        '''Test thin on a bar of width 3'''
        image = np.zeros((10,10), bool)
        image[3:6,2:8] = True
        expected = np.zeros((10,10), bool)
        expected[4,3:7] = True
        result = morph.thin(expected,iterations = None)
        self.assertTrue(np.all(result==expected))
    
    def test_02_01_random(self):
        '''A random image should preserve its Euler number'''
        np.random.seed(0)
        for i in range(20):
            image = np.random.uniform(size=(100,100)) < .1+float(i)/30.
            expected_euler_number = morph.euler_number(image)
            result = morph.thin(image)
            euler_number = morph.euler_number(result)
            if euler_number != expected_euler_number:
                from scipy.io.matlab import savemat
                savemat("c:\\temp\\euler.mat", 
                        {"orig":image, 
                         "orig_euler":np.array([expected_euler_number]),
                         "result":result,
                         "result_euler":np.array([euler_number]) },
                         False, "5", True)
            self.assertTrue(expected_euler_number == euler_number)
    
    def test_03_01_labels(self):
        '''Thin a labeled image'''
        image = np.zeros((10,10), int)
        #
        # This is two touching bars
        #
        image[3:6,2:8] = 1
        image[6:9,2:8] = 2
        expected = np.zeros((10,10),int)
        expected[4,3:7] = 1
        expected[7,3:7] = 2
        result = morph.thin(expected,iterations = None)
        self.assertTrue(np.all(result==expected))

class TestTableLookup(unittest.TestCase):
    def test_01_01_all_centers(self):
        '''Test table lookup at pixels off of the edge'''
        image = np.zeros((512*3+2,5),bool)
        for i in range(512):
            pattern = morph.pattern_of(i)
            image[i*3+1:i*3+4,1:4] = pattern
        table = np.arange(512)
        table[511] = 0 # do this to force using the normal mechanism
        index = morph.table_lookup(image, table, False, 1)
        self.assertTrue(np.all(index[2::3,2] == table))
    
    def test_01_02_all_corners(self):
        '''Test table lookup at the corners of the image'''
        np.random.seed(0)
        for iteration in range(100):
            table = np.random.uniform(size=512) > .5
            for p00 in (False,True):
                for p01 in (False, True):
                    for p10 in (False, True):
                        for p11 in (False,True):
                            image = np.array([[False,False,False,False,False,False],
                                              [False,p00,  p01,  p00,  p01,  False],
                                              [False,p10,  p11,  p10,  p11,  False],
                                              [False,p00,  p01,  p00,  p01,  False],
                                              [False,p10,  p11,  p10,  p11,  False],
                                              [False,False,False,False,False,False]])
                            expected = morph.table_lookup(image,table,False,1)[1:-1,1:-1]
                            result = morph.table_lookup(image[1:-1,1:-1],table,False,1)
                            self.assertTrue(np.all(result==expected),
                                            "Failure case:\n%7s,%s\n%7s,%s"%
                                            (p00,p01,p10,p11))
    
    def test_01_03_all_edges(self):
        '''Test table lookup along the edges of the image'''
        image = np.zeros((32*3+2,6),bool)
        np.random.seed(0)
        for iteration in range(100):
            table = np.random.uniform(size=512) > .5
            for i in range(32):
                pattern = morph.pattern_of(i)
                image[i*3+1:i*3+4,1:3] = pattern[:,:2]
                image[i*3+1:i*3+4,3:5] = pattern[:,:2]
            for im in (image,image.transpose()):
                expected = morph.table_lookup(im,table,False, 1)[1:-1,1:-1]
                result = morph.table_lookup(im[1:-1,1:-1],table,False,1)
                self.assertTrue(np.all(result==expected))
         
class TestBlock(unittest.TestCase):
    def test_01_01_one_block(self):
        labels, indexes = morph.block((10,10),(10,10))
        self.assertEqual(len(indexes),1)
        self.assertEqual(indexes[0],0)
        self.assertTrue(np.all(labels==0))
        self.assertEqual(labels.shape,(10,10))
    
    def test_01_02_six_blocks(self):
        labels, indexes = morph.block((10,15),(5,5))
        self.assertEqual(len(indexes),6)
        self.assertEqual(labels.shape, (10,15))
        i,j = np.mgrid[0:10,0:15]
        self.assertTrue(np.all(labels == (i / 5).astype(int)*3 + (j/5).astype(int)))

    def test_01_03_big_blocks(self):
        labels, indexes = morph.block((10,10),(20,20))
        self.assertEqual(len(indexes),1)
        self.assertEqual(indexes[0],0)
        self.assertTrue(np.all(labels==0))
        self.assertEqual(labels.shape,(10,10))

    def test_01_04_small_blocks(self):
        labels, indexes = morph.block((100,100),(2,4))
        self.assertEqual(len(indexes), 1250)
        i,j = np.mgrid[0:100,0:100]
        i = (i / 2).astype(int)
        j = (j / 4).astype(int)
        expected = i * 25 + j
        self.assertTrue(np.all(labels == expected))

class TestNeighbors(unittest.TestCase):
    def test_00_00_zeros(self):
        labels = np.zeros((10,10),int)
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        self.assertEqual(len(v_counts), 0)
        self.assertEqual(len(v_indexes), 0)
        self.assertEqual(len(v_neighbors), 0)
    
    def test_01_01_no_touch(self):
        labels = np.zeros((10,10),int)
        labels[2,2] = 1
        labels[7,7] = 2
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        self.assertEqual(len(v_counts), 2)
        self.assertEqual(v_counts[0], 0)
        self.assertEqual(v_counts[1], 0)
    
    def test_01_02_touch(self):
        labels = np.zeros((10,10),int)
        labels[2,2:5] = 1
        labels[3,2:5] = 2
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        self.assertEqual(len(v_counts), 2)
        self.assertEqual(v_counts[0], 1)
        self.assertEqual(v_neighbors[v_indexes[0]], 2)
        self.assertEqual(v_counts[1], 1)
        self.assertEqual(v_neighbors[v_indexes[1]], 1)
    
    def test_01_03_complex(self):
        labels = np.array([[1,1,2,2],
                           [2,2,2,3],
                           [4,3,3,3],
                           [5,6,3,3],
                           [0,7,8,9]])
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        self.assertEqual(len(v_counts), 9)
        for i, neighbors in ((1,[2]),
                             (2,[1,3,4]),
                             (3,[2,4,5,6,7,8,9]),
                             (4,[2,3,5,6]),
                             (5,[3,4,6,7]),
                             (6,[3,4,5,7,8]),
                             (7,[3,5,6,8]),
                             (8,[3,6,7,9]),
                             (9,[3,8])):
            i_neighbors = v_neighbors[v_indexes[i-1]:v_indexes[i-1]+v_counts[i-1]]
            self.assertTrue(np.all(i_neighbors == np.array(neighbors)))

class TestColor(unittest.TestCase):
    def test_01_01_color_zeros(self):
        '''Color a labels matrix of all zeros'''
        labels = np.zeros((10,10), int)
        colors = morph.color_labels(labels)
        self.assertTrue(np.all(colors==0))
    
    def test_01_02_color_ones(self):
        '''color a labels matrix of all ones'''
        labels = np.ones((10,10), int)
        colors = morph.color_labels(labels)
        self.assertTrue(np.all(colors==1))

    def test_01_03_color_complex(self):
        '''Create a bunch of shapes using Voroni cells and color them'''
        np.random.seed(0)
        mask = np.random.uniform(size=(100,100)) < .1
        labels,count = scind.label(mask, np.ones((3,3),bool))
        distances,(i,j) = scind.distance_transform_edt(~mask, 
                                                       return_indices = True)
        labels = labels[i,j]
        colors = morph.color_labels(labels)
        l00 = labels[1:-2,1:-2]
        c00 = colors[1:-2,1:-2]
        for i,j in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            lij = labels[1+i:i-2,1+j:j-2]
            cij = colors[1+i:i-2,1+j:j-2]
            self.assertTrue(np.all((l00 == lij) | (c00 != cij)))
            
    def test_02_01_color_127(self):
        '''Color 127 labels stored in a int8 array
        
        Regression test of img-1099
        '''
        # Create 127 labels
        labels = np.zeros((32,16), np.int8)
        i,j = np.mgrid[0:32, 0:16]
        mask = (i % 2 > 0) & (j % 2 > 0)
        labels[mask] = np.arange(np.sum(mask))
        colors = morph.color_labels(labels)
        self.assertTrue(np.all(colors[labels==0] == 0))
        self.assertTrue(np.all(colors[labels!=0] == 1))
            
class TestSkeletonizeLabels(unittest.TestCase):
    def test_01_01_skeletonize_complex(self):
        '''Skeletonize a complex field of shapes and check each individually'''
        np.random.seed(0)
        mask = np.random.uniform(size=(100,100)) < .1
        labels,count = scind.label(mask, np.ones((3,3),bool))
        distances,(i,j) = scind.distance_transform_edt(~mask, 
                                                       return_indices = True)
        labels = labels[i,j]
        skel = morph.skeletonize_labels(labels)
        for i in range(1,count+1,10):
            mask = labels == i
            skel_test = morph.skeletonize(mask)
            self.assertTrue(np.all(skel[skel_test] == i))
            self.assertTrue(np.all(skel[~skel_test] != i))

class TestAssociateByDistance(unittest.TestCase):
    def test_01_01_zeros(self):
        '''Test two label matrices with nothing in them'''
        result = morph.associate_by_distance(np.zeros((10,10),int),
                                             np.zeros((10,10),int), 0)
        self.assertEqual(result.shape[0], 0)
    
    def test_01_02_one_zero(self):
        '''Test a labels matrix with objects against one without'''
        result = morph.associate_by_distance(np.ones((10,10),int),
                                             np.zeros((10,10),int), 0)
        self.assertEqual(result.shape[0], 0)
    
    def test_02_01_point_in_square(self):
        '''Test a single point in a square'''
        #
        # Point is a special case - only one point in its convex hull
        #
        l1 = np.zeros((10,10),int)
        l1[1:5,1:5] = 1
        l1[5:9,5:9] = 2
        l2 = np.zeros((10,10),int)
        l2[2,3] = 3
        l2[2,9] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],3)
    
    def test_02_02_line_in_square(self):
        '''Test a line in a square'''
        l1 = np.zeros((10,10),int)
        l1[1:5,1:5] = 1
        l1[5:9,5:9] = 2
        l2 = np.zeros((10,10),int)
        l2[2,2:5] = 3
        l2[2,6:9] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],3)
    
    def test_03_01_overlap(self):
        '''Test a square overlapped by four other squares'''
        
        l1 = np.zeros((20,20),int)
        l1[5:16,5:16] = 1
        l2 = np.zeros((20,20),int)
        l2[1:6,1:6] = 1
        l2[1:6,14:19] = 2
        l2[14:19,1:6] = 3
        l2[14:19,14:19] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        self.assertEqual(result.shape[0],4)
        self.assertTrue(np.all(result[:,0]==1))
        self.assertTrue(all([x in result[:,1] for x in range(1,5)]))
    
    def test_03_02_touching(self):
        '''Test two objects touching at one point'''
        l1 = np.zeros((10,10), int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,10), int)
        l2[5:9,5:9] = 1
        result = morph.associate_by_distance(l1, l2, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],1)
    
    def test_04_01_distance_square(self):
        '''Test two squares separated by a distance'''
        l1 = np.zeros((10,20),int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,20),int)
        l2[3:6,10:16] = 1
        result = morph.associate_by_distance(l1,l2, 4)
        self.assertEqual(result.shape[0],0)
        result = morph.associate_by_distance(l1,l2, 5)
        self.assertEqual(result.shape[0],1)
    
    def test_04_02_distance_triangle(self):
        '''Test a triangle and a square (edge to point)'''
        l1 = np.zeros((10,20),int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,20),int)
        l2[4,10] = 1
        l2[3:6,11] = 1
        l2[2:7,12] = 1
        result = morph.associate_by_distance(l1,l2, 4)
        self.assertEqual(result.shape[0],0)
        result = morph.associate_by_distance(l1,l2, 5)
        self.assertEqual(result.shape[0],1)

class TestDistanceToEdge(unittest.TestCase):
    '''Test distance_to_edge'''
    def test_01_01_zeros(self):
        '''Test distance_to_edge with a matrix of zeros'''
        result = morph.distance_to_edge(np.zeros((10,10),int))
        self.assertTrue(np.all(result == 0))
    
    def test_01_02_square(self):
        '''Test distance_to_edge with a 3x3 square'''
        labels = np.zeros((10,10), int)
        labels[3:6,3:6] = 1
        expected = np.zeros((10,10))
        expected[3:6,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        result = morph.distance_to_edge(labels)
        self.assertTrue(np.all(result == expected))
    
    def test_01_03_touching(self):
        '''Test distance_to_edge when two objects touch each other'''
        labels = np.zeros((10,10), int)
        labels[3:6,3:6] = 1
        labels[6:9,3:6] = 2
        expected = np.zeros((10,10))
        expected[3:6,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        expected[6:9,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        result = morph.distance_to_edge(labels)
        self.assertTrue(np.all(result == expected))

class TestGreyReconstruction(unittest.TestCase):
    '''Test grey_reconstruction'''
    def test_01_01_zeros(self):
        '''Test grey_reconstruction with image and mask of zeros'''
        self.assertTrue(np.all(morph.grey_reconstruction(np.zeros((5,7)),
                                                         np.zeros((5,7))) == 0))
    
    def test_01_02_image_equals_mask(self):
        '''Test grey_reconstruction where the image and mask are the same'''
        self.assertTrue(np.all(morph.grey_reconstruction(np.ones((7,5)),
                                                         np.ones((7,5))) == 1))
    
    def test_01_03_image_less_than_mask(self):
        '''Test grey_reconstruction where the image is uniform and less than mask'''
        image = np.ones((5,5))
        mask = np.ones((5,5)) * 2
        self.assertTrue(np.all(morph.grey_reconstruction(image,mask) == 1))
    
    def test_01_04_one_image_peak(self):
        '''Test grey_reconstruction with one peak pixel'''
        image = np.ones((5,5))
        image[2,2] = 2
        mask = np.ones((5,5)) * 3
        self.assertTrue(np.all(morph.grey_reconstruction(image,mask) == 2))
    
    def test_01_05_two_image_peaks(self):
        '''Test grey_reconstruction with two peak pixels isolated by the mask'''
        image = np.array([[1,1,1,1,1,1,1,1],
                          [1,2,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,3,1],
                          [1,1,1,1,1,1,1,1]])
        
        mask = np.array([[4,4,4,1,1,1,1,1],
                         [4,4,4,1,1,1,1,1],
                         [4,4,4,1,1,1,1,1],
                         [1,1,1,1,1,4,4,4],
                         [1,1,1,1,1,4,4,4],
                         [1,1,1,1,1,4,4,4]])

        expected = np.array([[2,2,2,1,1,1,1,1],
                             [2,2,2,1,1,1,1,1],
                             [2,2,2,1,1,1,1,1],
                             [1,1,1,1,1,3,3,3],
                             [1,1,1,1,1,3,3,3],
                             [1,1,1,1,1,3,3,3]])
        self.assertTrue(np.all(morph.grey_reconstruction(image,mask) ==
                               expected))
    
    def test_02_01_zero_image_one_mask(self):
        '''Test grey_reconstruction with an image of all zeros and a mask that's not'''
        result = morph.grey_reconstruction(np.zeros((10,10)), np.ones((10,10)))
        self.assertTrue(np.all(result == 0))
        
class TestGetLinePts(unittest.TestCase):
    def test_01_01_no_pts(self):
        '''Can we call get_line_pts with zero-length vectors?'''
        i0, j0, i1, j1 = [np.zeros((0,))] * 4
        index, count, i, j = morph.get_line_pts(i0, j0, i1, j1)
        self.assertEqual(len(index), 0)
        self.assertEqual(len(count), 0)
        self.assertEqual(len(i), 0)
        self.assertEqual(len(j), 0)
    
    def test_01_02_horizontal_line(self):
        index, count, i, j = morph.get_line_pts([0],[0],[0],[10])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 11)
        self.assertTrue(np.all(i==0))
        self.assertTrue(np.all(j==np.arange(11)))
    
    def test_01_03_vertical_line(self):
        index, count, i, j = morph.get_line_pts([0],[0],[10],[0])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 11)
        self.assertTrue(np.all(j==0))
        self.assertTrue(np.all(i==np.arange(11)))
    
    def test_01_04_diagonal_line(self):
        index, count, i, j = morph.get_line_pts([0],[0],[10],[10])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 11)
        self.assertTrue(np.all(j==np.arange(11)))
        self.assertTrue(np.all(i==np.arange(11)))
        
    def test_01_05_antidiagonal_line(self):
        index, count, i, j = morph.get_line_pts([0],[0],[10],[-10])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 11)
        self.assertTrue(np.all(j==-np.arange(11)))
        self.assertTrue(np.all(i==np.arange(11)))
        
    def test_01_06_single_point(self):
        index, count, i, j = morph.get_line_pts([0],[0],[0],[0])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 1)
        self.assertEqual(i[0], 0)
        self.assertEqual(j[0], 0)
        
    def test_02_01_test_many(self):
        np.random.seed(0)
        n = 100
        i0,i1,j0,j1 = (np.random.uniform(size=(4,n))*100).astype(int)
        index, count, i_out, j_out = morph.get_line_pts(i0, j0, i1, j1)
        #
        # Run the Bresenham algorithm on each of the points manually
        #
        for idx in range(n):
            diff_i = abs(i1[idx]-i0[idx])
            diff_j = abs(j1[idx]-j0[idx])
            i = i0[idx]
            j = j0[idx]
            self.assertTrue(count[idx] > 0)
            self.assertEqual(i_out[index[idx]], i)
            self.assertEqual(j_out[index[idx]], j)
            step_i = (i1[idx] > i0[idx] and 1) or -1
            step_j = (j1[idx] > j0[idx] and 1) or -1
            pt_idx = 0
            if diff_j > diff_i:
                # J varies fastest, do i before j
                remainder = diff_i*2 - diff_j
                while j != j1[idx]:
                    pt_idx += 1
                    self.assertTrue(count[idx] > pt_idx)
                    if remainder >= 0:
                        i += step_i
                        remainder -= diff_j*2
                    j += step_j
                    remainder += diff_i*2
                    self.assertEqual(i_out[index[idx]+pt_idx], i)
                    self.assertEqual(j_out[index[idx]+pt_idx], j)
            else:
                remainder = diff_j*2 - diff_i
                while i != i1[idx]:
                    pt_idx += 1
                    self.assertTrue(count[idx] > pt_idx)
                    if remainder >= 0:
                        j += step_j
                        remainder -= diff_i*2
                    i += step_i
                    remainder += diff_j*2
                    self.assertEqual(j_out[index[idx]+pt_idx], j)
                    self.assertEqual(i_out[index[idx]+pt_idx], i)

class TestAllConnectedComponents(unittest.TestCase):
    def test_01_01_no_edges(self):
        result = morph.all_connected_components(np.array([], int), np.array([], int))
        self.assertEqual(len(result), 0)
        
    def test_01_02_one_component(self):
        result = morph.all_connected_components(np.array([0]), np.array([0]))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0], 0)
        
    def test_01_03_two_components(self):
        result = morph.all_connected_components(np.array([0,1]), 
                                                np.array([0,1]))
        self.assertEqual(len(result),2)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)
        
    def test_01_04_one_connection(self):
        result = morph.all_connected_components(np.array([0,1,2]),
                                                np.array([0,2,1]))
        self.assertEqual(len(result),3)
        self.assertTrue(np.all(result == np.array([0,1,1])))
        
    def test_01_05_components_can_label(self):
        #
        # all_connected_components can be used to label a matrix
        #
        np.random.seed(0)
        for d in ((10,12),(100,102)):
            mask = np.random.uniform(size=d) < .2
            mask[-1,-1] = True
            #
            # Just do 4-connectivity
            #
            labels, count = scind.label(mask)
            i,j = np.mgrid[0:d[0],0:d[1]]
            connected_top = (i > 0) & mask[i,j] & mask[i-1,j]
            idx = np.arange(np.prod(d))
            idx.shape = d
            connected_top_j = idx[connected_top] - d[1]
            
            connected_bottom = (i < d[0]-1) & mask[i,j] & mask[(i+1) % d[0],j]
            connected_bottom_j = idx[connected_bottom] + d[1]
            
            connected_left = (j > 0) & mask[i,j] & mask[i,j-1]
            connected_left_j = idx[connected_left] - 1
            
            connected_right = (j < d[1]-1) & mask[i,j] & mask[i,(j+1) % d[1]]
            connected_right_j = idx[connected_right] + 1
            
            i = np.hstack((idx[mask],
                           idx[connected_top],
                           idx[connected_bottom],
                           idx[connected_left],
                           idx[connected_right]))
            j = np.hstack((idx[mask], connected_top_j, connected_bottom_j,
                           connected_left_j, connected_right_j))
            result = morph.all_connected_components(i,j)
            self.assertEqual(len(result), np.prod(d))
            result.shape = d
            result[mask] += 1
            result[~mask] = 0
            #
            # Correlate the labels with the result
            #
            coo = scipy.sparse.coo_matrix((np.ones(np.prod(d)),
                                           (labels.flatten(),
                                            result.flatten())))
            corr = coo.toarray()
            #
            # Make sure there's either no or one hit per label association
            #
            self.assertTrue(np.all(np.sum(corr != 0,0) <= 1))
            self.assertTrue(np.all(np.sum(corr != 0,1) <= 1))
            
class TestBranchings(unittest.TestCase):
    def test_00_00_zeros(self):
        self.assertTrue(np.all(morph.branchings(np.zeros((10,11), bool)) == 0))
        
    def test_01_01_endpoint(self):
        image = np.zeros((10,11), bool)
        image[5,5:] = True
        self.assertEqual(morph.branchings(image)[5,5], 1)
        
    def test_01_02_line(self):
        image = np.zeros((10,11), bool)
        image[1:9, 5] = True
        self.assertTrue(np.all(morph.branchings(image)[2:8,5] == 2))
        
    def test_01_03_vee(self):
        image = np.zeros((11,11), bool)
        i,j = np.mgrid[-5:6,-5:6]
        image[-i == abs(j)] = True
        image[(j==0) & (i > 0)] = True
        self.assertTrue(morph.branchings(image)[5,5] == 3)
        
    def test_01_04_quadrabranch(self):
        image = np.zeros((11,11), bool)
        i,j = np.mgrid[-5:6,-5:6]
        image[abs(i) == abs(j)] = True
        self.assertTrue(morph.branchings(image)[5,5] == 4)
        
class TestLabelSkeleton(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Label a skeleton containing nothing'''
        skeleton = np.zeros((20,10), bool)
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 0)
        self.assertTrue(np.all(result == 0))
        
    def test_01_01_point(self):
        '''Label a skeleton consisting of a single point'''
        skeleton = np.zeros((20,10), bool)
        skeleton[5,5] = True
        expected = np.zeros((20,10), int)
        expected[5,5] = 1
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 1)
        self.assertTrue(np.all(result == expected))
        
    def test_01_02_line(self):
        '''Label a skeleton that's a line'''
        skeleton = np.zeros((20,10), bool)
        skeleton[5:15, 5] = True
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 1)
        self.assertTrue(np.all(result[skeleton] == 1))
        self.assertTrue(np.all(result[~skeleton] == 0))
        
    def test_01_03_branch(self):
        '''Label a skeleton that has a branchpoint'''
        skeleton = np.zeros((21,11), bool)
        i,j = np.mgrid[-10:11,-5:6]
        #
        # Looks like this:
        #  .   .
        #   . .
        #    .
        #    .
        skeleton[(i < 0) & (np.abs(i) == np.abs(j))] = True
        skeleton[(i >= 0) & (j == 0)] = True
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 4)
        self.assertTrue(np.all(result[~skeleton] == 0))
        self.assertTrue(np.all(result[skeleton] > 0))
        self.assertEqual(result[10,5], 1)
        v1 = result[5,0]
        v2 = result[5,-1]
        v3 = result[-1, 5]
        self.assertEqual(len(np.unique((v1, v2, v3))), 3)
        self.assertTrue(np.all(result[(i < 0) & (i==j)] == v1))
        self.assertTrue(np.all(result[(i < 0) & (i==-j)] == v2))
        self.assertTrue(np.all(result[(i > 0) & (j == 0)] == v3))
        
    def test_02_01_branch_and_edge(self):
        '''A branchpoint meeting an edge at two points'''
        
        expected = np.array(((2,0,0,0,0,1),
                             (0,2,0,0,1,0),
                             (0,0,3,1,0,0),
                             (0,0,4,0,0,0),
                             (0,4,0,0,0,0),
                             (4,0,0,0,0,0)))
        skeleton = expected > 0
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 4)
        self.assertTrue(np.all(result[~skeleton] == 0))
        self.assertEqual(len(np.unique(result)), 5)
        self.assertEqual(np.max(result), 4)
        self.assertEqual(np.min(result), 0)
        for i in range(1,5):
            self.assertEqual(len(np.unique(result[expected == i])), 1)

    def test_02_02_four_edges_meet(self):
        '''Check the odd case of four edges meeting at a square
        
        The shape is something like this:
        
        .    .
         .  .
          ..
          ..
         .  .
        .    .
        None of the points above are branchpoints - they're sort of
        half-branchpoints.
        '''
        i,j = np.mgrid[-10:10,-10:10]
        i[i<0] += 1
        j[j<0] += 1
        skeleton=np.abs(i) == np.abs(j)
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 4)
        self.assertTrue(np.all(result[~skeleton]==0))
        self.assertEqual(np.max(result), 4)
        self.assertEqual(np.min(result), 0)
        self.assertEqual(len(np.unique(result)), 5)
        for im in (-1, 1):
            for jm in (-1, 1):
                self.assertEqual(len(np.unique(result[(i*im == j*jm) & 
                                                      (i*im > 0) &
                                                      (j*jm > 0)])), 1)
                
class TestPairwisePermutations(unittest.TestCase):
    def test_00_00_empty(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([]), np.array([]))
        for x in (i, j1, j2):
            self.assertEqual(len(x), 0)
            
    def test_00_01_no_permutations_of_one(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([4]), np.array([3]))
        for x in (i, j1, j2):
            self.assertEqual(len(x), 0)

    def test_01_01_two(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([4,4]), np.array([9,3]))
        for x, v in ((i, 4), (j1, 3), (j2, 9)):
            self.assertEqual(len(x), 1)
            self.assertEqual(x[0], v)
    
    def test_01_02_many(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([7,7,7,5,5,5,5,9,9,9,9,9,9]),
                                              np.array([1,3,2,4,5,8,6,1,2,3,4,5,6]))
        for x, v in (
            (i,  np.array([5,5,5,5,5,5,7,7,7,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9])),
            (j1, np.array([4,4,4,5,5,6,1,1,2,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5])),
            (j2, np.array([5,6,8,6,8,8,2,3,3,2,3,4,5,6,3,4,5,6,4,5,6,5,6,6]))):
            self.assertEqual(len(x), len(v))
            self.assertTrue(np.all(x == v))

class TestIsLocalMaximum(unittest.TestCase):
    def test_00_00_empty(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(~ result))
        
    def test_01_01_one_point(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        labels[5,5] = 1
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == (labels == 1)))
        
    def test_01_02_adjacent_and_same(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5:6] = 1
        labels[5,5:6] = 1
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == (labels == 1)))
        
    def test_01_03_adjacent_and_different(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,6] = .5
        labels[5,5:6] = 1
        expected = (image == 1)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == expected))
        
    def test_01_04_not_adjacent_and_different(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,8] = .5
        labels[image > 0] = 1
        expected = (labels == 1)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == expected))
        
    def test_01_05_two_objects(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,15] = .5
        labels[5,5] = 1
        labels[5,15] = 2
        expected = (labels > 0)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == expected))

    def test_01_06_adjacent_different_objects(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,6] = .5
        labels[5,5] = 1
        labels[5,6] = 2
        expected = (labels > 0)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == expected))
        
    def test_02_01_four_quadrants(self):
        np.random.seed(21)
        image = np.random.uniform(size=(40,60))
        i,j = np.mgrid[0:40,0:60]
        labels = 1 + (i >= 20) + (j >= 30) * 2
        i,j = np.mgrid[-3:4,-3:4]
        footprint = (i*i + j*j <=9)
        expected = np.zeros(image.shape, float)
        for imin, imax in ((0, 20), (20, 40)):
            for jmin, jmax in ((0, 30), (30, 60)):
                expected[imin:imax,jmin:jmax] = scind.maximum_filter(
                    image[imin:imax, jmin:jmax], footprint = footprint)
        expected = (expected == image)
        result = morph.is_local_maximum(image, labels, footprint)
        self.assertTrue(np.all(result == expected))
        
    def test_03_01_disk_1(self):
        '''regression test of img-1194, footprint = [1]
        
        Test is_local_maximum when every point is a local maximum
        '''
        np.random.seed(31)
        image = np.random.uniform(size=(10,20))
        footprint = morph.strel_disk(.5)
        self.assertEqual(np.prod(footprint.shape), 1)
        self.assertEqual(footprint[0,0], 1)
        result = morph.is_local_maximum(image, np.ones((10,20)), footprint)
        self.assertTrue(np.all(result))
        

class TestAngularDistribution(unittest.TestCase):
    def test_00_00_angular_dist(self):
        np.random.seed(0)
        # random labels from 0 to 9
        labels = (np.random.uniform(0, 0.95, (1000, 1000)) * 10).astype(np.int)
        # filled square of 11 (NB: skipped 10)
        labels[200:300, 600:900] = 11
        angdist = morph.angular_distribution(labels)
        # 10 is an empty label
        assert np.all(angdist[9, :] == 0.0)
        # check approximation to chord ratio of filled rectangle (roughly 3.16)
        resolution = angdist.shape[1]
        angdist2 = angdist[-1, :resolution/2] + angdist[-1, resolution/2:]
        assert np.abs(3.16 - np.sqrt(angdist2.max() / angdist2.min())) < 0.05

class TestFeretDiameter(unittest.TestCase):
    def test_00_00_none(self):
        result = morph.feret_diameter(np.zeros((0,3)), np.zeros(0, int), [])
        self.assertEqual(len(result), 0)
        
    def test_00_01_point(self):
        min_result, max_result = morph.feret_diameter(
            np.array([[1, 0, 0]]),
            np.ones(1, int), [1])
        self.assertEqual(len(min_result), 1)
        self.assertEqual(min_result[0], 0)
        self.assertEqual(len(max_result), 1)
        self.assertEqual(max_result[0], 0)
        
    def test_01_02_line(self):
        min_result, max_result = morph.feret_diameter(
            np.array([[1, 0, 0], [1, 1, 1]]),
            np.array([2], int), [1])
        self.assertEqual(len(min_result), 1)
        self.assertEqual(min_result[0], 0)
        self.assertEqual(len(max_result), 1)
        self.assertEqual(max_result[0], np.sqrt(2))
        
    def test_01_03_single(self):
        r = np.random.RandomState()
        r.seed(204)
        niterations = 100
        iii = r.randint(0, 100, size=(20 * niterations))
        jjj = r.randint(0, 100, size=(20 * niterations))
        for iteration in range(100):
            ii = iii[(iteration * 20):((iteration + 1) * 20)]
            jj = jjj[(iteration * 20):((iteration + 1) * 20)]
            chulls, counts = morph.convex_hull_ijv(
                np.column_stack((ii, jj, np.ones(20, int))), [1])
            min_result, max_result = morph.feret_diameter(chulls, counts, [1])
            self.assertEqual(len(min_result), 1)
            distances = np.sqrt(
                ((ii[:,np.newaxis] - ii[np.newaxis,:]) ** 2 +
                 (jj[:,np.newaxis] - jj[np.newaxis,:]) ** 2).astype(float))
            expected = np.max(distances)
            if abs(max_result - expected) > .000001:
                a0,a1 = np.argwhere(distances == expected)[0]
                self.assertAlmostEqual(
                    max_result[0], expected,
                    msg = "Expected %f, got %f, antipodes are %d,%d and %d,%d" %
                (expected, result, ii[a0], jj[a0], ii[a1], jj[a1]))
            #
            # Do a 180 degree sweep, measuring
            # the Feret diameter at each angle. Stupid but an independent test.
            #
            # Draw a line segment from the origin to a point at the given
            # angle from the horizontal axis
            #
            angles = np.pi * np.arange(20).astype(float) / 20.0
            i = -np.sin(angles)
            j = np.cos(angles)
            chull_idx, angle_idx = np.mgrid[0:counts[0],0:20]
            #
            # Compose a list of all vertices on the convex hull and all lines
            #
            v = chulls[chull_idx.ravel(),1:]
            pt1 = np.zeros((20 * counts[0], 2))
            pt2 = np.column_stack([i[angle_idx.ravel()], j[angle_idx.ravel()]])
            #
            # For angles from 90 to 180, the parallel line has to be sort of
            # at negative infinity instead of zero to keep all points on
            # the same side
            #
            pt1[angle_idx.ravel() < 10,1] -= 200
            pt2[angle_idx.ravel() < 10,1] -= 200
            pt1[angle_idx.ravel() >= 10,0] += 200
            pt2[angle_idx.ravel() >= 10,0] += 200
            distances = np.sqrt(morph.distance2_to_line(v, pt1, pt2))
            distances.shape = (counts[0], 20)
            dmin = np.min(distances, 0)
            dmax = np.max(distances, 0)
            expected_min = np.min(dmax - dmin)
            self.assertTrue(min_result[0] <= expected_min)
            
    def test_02_01_multiple_objects(self):
        r = np.random.RandomState()
        r.seed(204)
        niterations = 100
        ii = r.randint(0, 100, size=(20 * niterations))
        jj = r.randint(0, 100, size=(20 * niterations))
        vv = np.hstack([np.ones(20) * i for i in range(1,niterations+1)])
        indexes = np.arange(1, niterations+1)
        chulls, counts = morph.convex_hull_ijv(
            np.column_stack((ii, jj, vv)), indexes)
        min_result, max_result = morph.feret_diameter(chulls, counts, indexes)
        self.assertEqual(len(max_result), niterations)
        for i in range(niterations):
            #
            # Make sure values are same as single (validated) case.
            #
            iii = ii[(20*i):(20*(i+1))]
            jjj = jj[(20*i):(20*(i+1))]
            chulls, counts = morph.convex_hull_ijv(
                np.column_stack((iii, jjj, np.ones(len(iii), int))), [1])
            expected_min, expected_max = morph.feret_diameter(chulls, counts, [1])
            self.assertAlmostEqual(expected_min[0], min_result[i])
            self.assertAlmostEqual(expected_max[0], max_result[i])


class TestIsObtuse(unittest.TestCase):
    def test_00_00_empty(self):
        result = morph.is_obtuse(np.zeros((0,2)),np.zeros((0,2)),np.zeros((0,2)))
        self.assertEqual(len(result), 0)
        
    def test_01_01_is_obtuse(self):
        result = morph.is_obtuse(np.array([[-1,1]]),
                                 np.array([[0,0]]),
                                 np.array([[1,0]]))
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0])
        
    def test_01_02_is_not_obtuse(self):
        result = morph.is_obtuse(np.array([[1,1]]),
                                 np.array([[0,0]]),
                                 np.array([[1,0]]))
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0])
        
    def test_01_03_many(self):
        r = np.random.RandomState()
        r.seed(13)
        p1 = np.random.uniform(size=(100,2))
        v = np.random.uniform(size=(100,2))
        p2 = np.random.uniform(size=(100,2))
        vp1 = np.sqrt(np.sum((v - p1) * (v - p1), 1))
        vp2 = np.sqrt(np.sum((v - p2) * (v - p2), 1))
        p1p2 = np.sqrt(np.sum((p1-p2) * (p1-p2), 1))
        # Law of cosines
        theta = np.arccos((vp1**2 + vp2**2 - p1p2 **2) / (2 * vp1 * vp2))
        result = morph.is_obtuse(p1, v, p2)
        is_obtuse = theta > np.pi / 2
        np.testing.assert_array_equal(result, is_obtuse)
        
class TestSingleShortestPaths(unittest.TestCase):
    def test_00_00_one_node(self):
        p, c = morph.single_shortest_paths(0, np.zeros((1,1)))
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0], 0)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0], 0)
        
    def test_01_01_two_nodes(self):
        p, c = morph.single_shortest_paths(0, np.array([[0,1],[1,0]]))
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0], 0)
        self.assertEqual(p[1], 0)
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0], 0)
        self.assertEqual(c[1], 1)
        
    def test_01_02_two_nodes_backwards(self):
        p, c = morph.single_shortest_paths(1, np.array([[0,1],[1,0]]))
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0], 1)
        self.assertEqual(p[1], 1)
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0], 1)
        self.assertEqual(c[1], 0)
        
    def test_01_03_5x5(self):
        # All paths from 0 to 4
        all_permutations = np.array([
            [ 0, 0, 0, 0, 4],
            [ 0, 0, 0, 1, 4],
            [ 0, 0, 0, 2, 4],
            [ 0, 0, 0, 3, 4],
            [ 0, 0, 1, 2, 4],
            [ 0, 0, 1, 3, 4],
            [ 0, 0, 2, 1, 4],
            [ 0, 0, 2, 3, 4],
            [ 0, 0, 3, 1, 4],
            [ 0, 0, 3, 2, 4],
            [ 0, 1, 2, 3, 4],
            [ 0, 1, 3, 2, 4],
            [ 0, 2, 1, 3, 4],
            [ 0, 2, 3, 1, 4],
            [ 0, 3, 1, 2, 4],
            [ 0, 3, 2, 1, 4]
        ])
        r = np.random.RandomState()
        r.seed(13)
        for _ in range(1000):
            c = r.uniform(size=(5,5))
            c[np.arange(5), np.arange(5)] = 0
            steps = c[all_permutations[:, :-1],
                      all_permutations[:, 1:]]
            all_costs = np.sum(steps, 1)
            best_path = all_permutations[np.argmin(all_costs)]
            best_path = list(reversed(best_path[best_path != 0][:-1]))
            best_score = np.min(all_costs)
            paths, scores = morph.single_shortest_paths(0, c)
            self.assertEqual(scores[4], best_score)
            step_count = 0
            found_path = []
            i = 4
            while step_count != 5 and paths[i] != 0:
                i = paths[i]
                found_path.append(i)
                step_count += 1
            self.assertEqual(len(found_path), len(best_path))
            self.assertTrue(all([a == b for a,b in zip(found_path, best_path)]))
            
