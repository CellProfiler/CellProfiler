import unittest
import numpy as np
from cpmorphology import draw_line
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})
import _convex_hull as morph

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
        self.assertEqual(counts.shape[0], 2)
        np.testing.assert_equal(counts, 1)
        
        expected = np.array([[2,5,6],[1,2,3]])
        np.testing.assert_equal(result, expected)
    
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
                    draw_line(labels, p[k,:], p[(k+1)%ct,:], idx)
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

