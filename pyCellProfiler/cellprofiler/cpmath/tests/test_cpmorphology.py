"""test_cpmorphology - test the functions in cellprofiler.cpmath.cpmorphology

"""
__version__="$Revision$"

import unittest
import numpy
import scipy.ndimage
import scipy.misc

import cellprofiler.cpmath.cpmorphology as morph

class TestFillLabeledHoles(unittest.TestCase):
    def test_01_zeros(self):
        """A label matrix of all zeros has no hole"""
        image = numpy.zeros((10,10),dtype=int)
        output = morph.fill_labeled_holes(image)
        self.assertTrue(numpy.all(output==0))

    def test_02_object_without_holes(self):
        """The label matrix of a single object without holes has no hole"""
        image = numpy.zeros((10,10),dtype=int)
        image[3:6,3:6] = 1
        output = morph.fill_labeled_holes(image)
        self.assertTrue(numpy.all(output==image))
    
    def test_03_object_with_hole(self):
        image = numpy.zeros((20,20),dtype=int)
        image[5:15,5:15] = 1
        image[8:12,8:12] = 0
        output = morph.fill_labeled_holes(image)
        self.assertTrue(numpy.all(output[8:12,8:12] == 1))
        output[8:12,8:12] = 0 # unfill the hole again
        self.assertTrue(numpy.all(output==image))
    
    def test_04_holes_on_edges_are_not_holes(self):
        image = numpy.zeros((40,40),dtype=int)
        objects = (((15,25),(0,10),(18,22),(0,3)),
                   ((0,10),(15,25),(0,3),(18,22)),
                   ((15,25),(30,39),(18,22),(36,39)),
                   ((30,39),(15,25),(36,39),(18,22)))
        for idx,x in zip(range(1,len(objects)+1),objects):
            image[x[0][0]:x[0][1],x[1][0]:x[1][1]] = idx
            image[x[2][0]:x[2][1],x[3][0]:x[3][1]] = 0
        output = morph.fill_labeled_holes(image)
        for x in objects:
            self.assertTrue(numpy.all(output[x[2][0]:x[2][1],x[3][0]:x[3][1]]==0))
            output[x[2][0]:x[2][1],x[3][0]:x[3][1]] = 1
            self.assertTrue(numpy.all(output[x[0][0]:x[0][1],x[1][0]:x[1][1]]!=0))
    
    def test_05_lots_of_objects_with_holes(self):
        image = numpy.ones((1020,1020))
        for i in range(0,51):
            image[i*20:i*20+20,:]=0
            image[:,i*20:i*20+20]=0
        binary_image = scipy.ndimage.gaussian_gradient_magnitude(image,.5) > 0.1
        labeled_image,nobjects = scipy.ndimage.label(binary_image)
        output = morph.fill_labeled_holes(labeled_image)
        eroded_image = scipy.ndimage.binary_erosion(image>0,iterations=3)
        self.assertTrue(numpy.all(output[eroded_image] > 0))
    
    def test_06_regression_diamond(self):
        """Check filling the center of a diamond"""
        image = numpy.zeros((5,5),int)
        image[1,2]=1
        image[2,1]=1
        image[2,3]=1
        image[3,2]=1
        output = morph.fill_labeled_holes(image)
        where = numpy.argwhere(image!=output)
        self.assertEqual(len(where),1)
        self.assertEqual(where[0][0],2)
        self.assertEqual(where[0][1],2)
    
    def test_07_regression_nearby_holes(self):
        """Check filling an object with three holes"""
        image = numpy.array([[0,0,0,0,0,0,0,0,0,0,0,0],
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
        expec = numpy.array([[0,0,0,0,0,0,0,0,0,0,0,0],
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
        self.assertTrue(numpy.all(output==expec))
            
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
        ya = numpy.array(y,dtype=float).reshape((5,5))
        self.assertTrue(numpy.all(x==ya))
    
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
        ya = numpy.array(y,dtype=float).reshape((5,5))
        self.assertTrue(numpy.all(x==ya))

class TestBinaryShrink(unittest.TestCase):
    def test_01_zeros(self):
        """Shrink an empty array to itself"""
        input = numpy.zeros((10,10),dtype=bool)
        result = morph.binary_shrink(input,1)
        self.assertTrue(numpy.all(input==result))
    
    def test_02_cross(self):
        """Shrink a cross to a single point"""
        input = numpy.zeros((9,9),dtype=bool)
        input[4,:]=True
        input[:,4]=True
        result = morph.binary_shrink(input)
        where = numpy.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_03_x(self):
        input = numpy.zeros((9,9),dtype=bool)
        x,y = numpy.mgrid[-4:5,-4:5]
        input[x==y]=True
        input[x==-y]=True
        result = morph.binary_shrink(input)
        where = numpy.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_04_block(self):
        """A block should shrink to a point"""
        input = numpy.zeros((9,9), dtype=bool)
        input[3:6,3:6]=True
        result = morph.binary_shrink(input)
        where = numpy.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_05_hole(self):
        """A hole in a block should shrink to a ring"""
        input = numpy.zeros((19,19), dtype=bool)
        input[5:15,5:15]=True
        input[9,9]=False
        result = morph.binary_shrink(input)
        where = numpy.argwhere(result)
        self.assertTrue(len(where) > 1)
        self.assertFalse(result[9:9])

    def test_06_random_filled(self):
        """Shrink random blobs
        
        If you label a random binary image, then fill the holes,
        then shrink the result, each blob should shrink to a point
        """
        numpy.random.seed(0)
        input = numpy.random.uniform(size=(300,300)) > .8
        labels,nlabels = scipy.ndimage.label(input,numpy.ones((3,3),bool))
        filled_labels = morph.fill_labeled_holes(labels)
        input = filled_labels > 0
        result = morph.binary_shrink(input)
        my_sum = scipy.ndimage.sum(result.astype(int),filled_labels,range(nlabels+1))
        my_sum = numpy.array(my_sum)
        self.assertTrue(numpy.all(my_sum[1:] == 1))
        
class TestCpmaximum(unittest.TestCase):
    def test_01_zeros(self):
        input = numpy.zeros((10,10))
        output = morph.cpmaximum(input)
        self.assertTrue(numpy.all(output==input))
    
    def test_01_ones(self):
        input = numpy.ones((10,10))
        output = morph.cpmaximum(input)
        self.assertTrue(numpy.all(output==input))

    def test_02_center_point(self):
        input = numpy.zeros((9,9))
        input[4,4] = 1
        expected = numpy.zeros((9,9))
        expected[3:6,3:6] = 1
        structure = numpy.ones((3,3),dtype=bool)
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(numpy.all(output==expected))
    
    def test_03_corner_point(self):
        input = numpy.zeros((9,9))
        input[0,0]=1
        expected = numpy.zeros((9,9))
        expected[:2,:2]=1
        structure = numpy.ones((3,3),dtype=bool)
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(numpy.all(output==expected))

    def test_04_structure(self):
        input = numpy.zeros((9,9))
        input[0,0]=1
        input[4,4]=1
        structure = numpy.zeros((3,3),dtype=bool)
        structure[0,0]=1
        expected = numpy.zeros((9,9))
        expected[1,1]=1
        expected[5,5]=1
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(numpy.all(output[1:,1:]==expected[1:,1:]))

    def test_05_big_structure(self):
        big_disk = morph.strel_disk(10).astype(bool)
        input = numpy.zeros((1001,1001))
        input[500,500] = 1
        expected = numpy.zeros((1001,1001))
        expected[490:551,490:551][big_disk]=1
        output = morph.cpmaximum(input,big_disk)
        self.assertTrue(numpy.all(output == expected))

class TestRelabel(unittest.TestCase):
    def test_00_relabel_zeros(self):
        input = numpy.zeros((10,10),int)
        output,count = morph.relabel(input)
        self.assertTrue(numpy.all(input==output))
        self.assertEqual(count, 0)
    
    def test_01_relabel_one(self):
        input = numpy.zeros((10,10),int)
        input[3:6,3:6]=1
        output,count = morph.relabel(input)
        self.assertTrue(numpy.all(input==output))
        self.assertEqual(count,1)
    
    def test_02_relabel_two_to_one(self):
        input = numpy.zeros((10,10),int)
        input[3:6,3:6]=2
        output,count = morph.relabel(input)
        self.assertTrue(numpy.all((output==1)[input==2]))
        self.assertTrue(numpy.all((input==output)[input!=2]))
        self.assertEqual(count,1)
    
    def test_03_relabel_gap(self):
        input = numpy.zeros((20,20),int)
        input[3:6,3:6]=1
        input[3:6,12:15]=3
        output,count = morph.relabel(input)
        self.assertTrue(numpy.all((output==2)[input==3]))
        self.assertTrue(numpy.all((input==output)[input!=3]))
        self.assertEqual(count,2)

class TestConvexHull(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure convex_hull can handle an empty array"""
        result,counts = morph.convex_hull(numpy.zeros((10,10),int), [])
        self.assertEqual(numpy.product(result.shape),0)
        self.assertEqual(numpy.product(counts.shape),0)
    
    def test_01_01_zeros(self):
        """Make sure convex_hull can work if a label has no points"""
        result,counts = morph.convex_hull(numpy.zeros((10,10),int), [1])
        self.assertEqual(numpy.product(result.shape),0)
        self.assertEqual(numpy.product(counts.shape),1)
        self.assertEqual(counts[0],0)
    
    def test_01_02_point(self):
        """Make sure convex_hull can handle the degenerate case of one point"""
        labels = numpy.zeros((10,10),int)
        labels[4,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(result.shape,(1,3))
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],4)
        self.assertEqual(result[0,2],5)
        self.assertEqual(counts[0],1)
    
    def test_01_030_line(self):
        """Make sure convex_hull can handle the degenerate case of a line"""
        labels = numpy.zeros((10,10),int)
        labels[2:8,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],2)
        self.assertEqual(result.shape,(2,3))
        self.assertTrue(numpy.all(result[:,0]==1))
        self.assertTrue(result[0,1] in (2,7))
        self.assertTrue(result[1,1] in (2,7))
        self.assertTrue(numpy.all(result[:,2]==5))
    
    def test_01_031_odd_line(self):
        """Make sure convex_hull can handle the degenerate case of a line with odd length
        
        This is a regression test: the line has a point in the center if
        it's odd and the sign of the difference of that point is zero
        which causes it to be included in the hull.
        """
        labels = numpy.zeros((10,10),int)
        labels[2:7,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],2)
        self.assertEqual(result.shape,(2,3))
        self.assertTrue(numpy.all(result[:,0]==1))
        self.assertTrue(result[0,1] in (2,6))
        self.assertTrue(result[1,1] in (2,6))
        self.assertTrue(numpy.all(result[:,2]==5))
    
    def test_01_04_square(self):
        """Make sure convex_hull can handle a square which is not degenerate"""
        labels = numpy.zeros((10,10),int)
        labels[2:7,3:8] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],4)
        order = numpy.lexsort((result[:,2], result[:,1]))
        result = result[order,:]
        expected = numpy.array([[1,2,3],
                                [1,2,7],
                                [1,6,3],
                                [1,6,7]])
        self.assertTrue(numpy.all(result==expected))
    
    def test_02_01_out_of_order(self):
        """Make sure convex_hull can handle out of order indices"""
        labels = numpy.zeros((10,10),int)
        labels[2,3] = 1
        labels[5,6] = 2
        result,counts = morph.convex_hull(labels,[2,1])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(numpy.all(counts==1))
        
        expected = numpy.array([[2,5,6],[1,2,3]])
        self.assertTrue(numpy.all(result == expected))
    
    def test_02_02_out_of_order(self):
        """Make sure convex_hull can handle out of order indices
        that require different #s of loop iterations"""
        
        labels = numpy.zeros((10,10),int)
        labels[2,3] = 1
        labels[1:7,4:8] = 2
        result,counts = morph.convex_hull(labels, [2,1])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(numpy.all(counts==(4,1)))
        self.assertEqual(result.shape,(5,3))
        order = numpy.lexsort((result[:,2],result[:,1],
                               numpy.array([0,2,1])[result[:,0]]))
        result = result[order,:]
        expected = numpy.array([[2,1,4],
                                [2,1,7],
                                [2,6,4],
                                [2,6,7],
                                [1,2,3]])
        self.assertTrue(numpy.all(result==expected))
    
    def test_02_03_two_squares(self):
        """Make sure convex_hull can handle two complex shapes"""
        labels = numpy.zeros((10,10),int)
        labels[1:5,3:7] = 1
        labels[6:10,1:7] = 2
        result,counts = morph.convex_hull(labels, [1,2])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(numpy.all(counts==(4,4)))
        order = numpy.lexsort((result[:,2],result[:,1],result[:,0]))
        result = result[order,:]
        expected = numpy.array([[1,1,3],[1,1,6],[1,4,3],[1,4,6],
                                [2,6,1],[2,6,6],[2,9,1],[2,9,6]])
        self.assertTrue(numpy.all(result==expected))
        
    def test_03_01_concave(self):
        """Make sure convex_hull handles a square with a concavity"""
        labels = numpy.zeros((10,10),int)
        labels[2:8,3:9] = 1
        labels[3:7,3] = 0
        labels[2:6,4] = 0
        labels[4:5,5] = 0
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],4)
        order = numpy.lexsort((result[:,2],result[:,1],result[:,0]))
        result = result[order,:]
        expected = numpy.array([[1,2,3],
                                [1,2,8],
                                [1,7,3],
                                [1,7,8]])
        self.assertTrue(numpy.all(result==expected))
        
    def test_04_01_regression(self):
        """The set of points given in this case yielded one in the interior"""
        numpy.random.seed(0)
        s = 10 # divide each image into this many mini-squares with a shape in each
        side = 250
        mini_side = side / s
        ct = 20
        labels = numpy.zeros((side,side),int)
        pts = numpy.zeros((s*s*ct,2),int)
        index = numpy.array(range(pts.shape[0])).astype(float)/float(ct)
        index = index.astype(int)
        idx = 0
        for i in range(0,side,mini_side):
            for j in range(0,side,mini_side):
                idx = idx+1
                # get ct+1 unique points
                p = numpy.random.uniform(low=0,high=mini_side,
                                         size=(ct+1,2)).astype(int)
                while True:
                    pu = numpy.unique(p[:,0]+p[:,1]*mini_side)
                    if pu.shape[0] == ct+1:
                        break
                    p[:pu.shape[0],0] = numpy.mod(pu,mini_side).astype(int)
                    p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                    p_size = (ct+1-pu.shape[0],2)
                    p[pu.shape[0],:] = numpy.random.uniform(low=0,
                                                            high=mini_side,
                                                            size=p_size)
                # Use the last point as the "center" and order
                # all of the other points according to their angles
                # to this "center"
                center = p[ct,:]
                v = p[:ct,:]-center
                angle = numpy.arctan2(v[:,0],v[:,1])
                order = numpy.lexsort((angle,))
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
        result,counts = morph.convex_hull(labels,numpy.array(range(100))+1)
        self.assertFalse(numpy.any(numpy.logical_and(result[:,1]==5,
                                                     result[:,2]==106)))
        
    
class TestMinimumEnclosingCircle(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure minimum_enclosing_circle can handle an empty array"""
        center,radius = morph.minimum_enclosing_circle(numpy.zeros((10,10),int), [])
        self.assertEqual(numpy.product(center.shape),0)
        self.assertEqual(numpy.product(radius.shape),0)
    
    def test_01_01_zeros(self):
        """Make sure minimum_enclosing_circle can work if a label has no points"""
        center,radius = morph.minimum_enclosing_circle(numpy.zeros((10,10),int), [1])
        self.assertEqual(center.shape,(1,2))
        self.assertEqual(numpy.product(radius.shape),1)
        self.assertEqual(radius[0],0)
    
    def test_01_02_point(self):
        """Make sure minimum_enclosing_circle can handle the degenerate case of one point"""
        labels = numpy.zeros((10,10),int)
        labels[4,5] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertEqual(center.shape,(1,2))
        self.assertEqual(radius.shape,(1,))
        self.assertTrue(numpy.all(center==numpy.array([(4,5)])))
        self.assertEqual(radius[0],0)
    
    def test_01_03_line(self):
        """Make sure minimum_enclosing_circle can handle the degenerate case of a line"""
        labels = numpy.zeros((10,10),int)
        labels[2:7,5] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertTrue(numpy.all(center==numpy.array([(4,5)])))
        self.assertEqual(radius[0],2)
    
    def test_01_04_square(self):
        """Make sure minimum_enclosing_circle can handle a square which is not degenerate"""
        labels = numpy.zeros((10,10),int)
        labels[2:7,3:8] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertTrue(numpy.all(center==numpy.array([(4,5)])))
        self.assertAlmostEqual(radius[0],numpy.sqrt(8))
    
    def test_02_01_out_of_order(self):
        """Make sure minimum_enclosing_circle can handle out of order indices"""
        labels = numpy.zeros((10,10),int)
        labels[2,3] = 1
        labels[5,6] = 2
        center,radius = morph.minimum_enclosing_circle(labels,[2,1])
        self.assertEqual(center.shape,(2,2))
        
        expected_center = numpy.array(((5,6),(2,3)))
        self.assertTrue(numpy.all(center == expected_center))
    
    def test_02_02_out_of_order(self):
        """Make sure minimum_enclosing_circle can handle out of order indices
        that require different #s of loop iterations"""
        
        labels = numpy.zeros((10,10),int)
        labels[2,3] = 1
        labels[1:6,4:9] = 2
        center,result = morph.minimum_enclosing_circle(labels, [2,1])
        expected_center = numpy.array(((3,6),(2,3)))
        self.assertTrue(numpy.all(center == expected_center))
    
    def test_03_01_random_polygons(self):
        """Test minimum_enclosing_circle on 250 random dodecagons"""
        numpy.random.seed(0)
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
            labels = numpy.zeros((side,side),int)
            pts = numpy.zeros((s*s*ct,2),int)
            index = numpy.array(range(pts.shape[0])).astype(float)/float(ct)
            index = index.astype(int)
            idx = 0
            for i in range(0,side,mini_side):
                for j in range(0,side,mini_side):
                    idx = idx+1
                    # get ct+1 unique points
                    p = numpy.random.uniform(low=0,high=mini_side,
                                             size=(ct+1,2)).astype(int)
                    while True:
                        pu = numpy.unique(p[:,0]+p[:,1]*mini_side)
                        if pu.shape[0] == ct+1:
                            break
                        p[:pu.shape[0],0] = numpy.mod(pu,mini_side).astype(int)
                        p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                        p_size = (ct+1-pu.shape[0],2)
                        p[pu.shape[0],:] = numpy.random.uniform(low=0,
                                                                high=mini_side,
                                                                size=p_size)
                    # Use the last point as the "center" and order
                    # all of the other points according to their angles
                    # to this "center"
                    center = p[ct,:]
                    v = p[:ct,:]-center
                    angle = numpy.arctan2(v[:,0],v[:,1])
                    order = numpy.lexsort((angle,))
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
                                                           numpy.array(range(s**2))+1)
            epsilon = .000001
            center_per_pt = center[index]
            radius_per_pt = radius[index]
            distance_from_center = numpy.sqrt(numpy.sum((pts.astype(float)-
                                                         center_per_pt)**2,1))
            #
            # All points must be within the enclosing circle
            #
            self.assertTrue(numpy.all(distance_from_center - epsilon < radius_per_pt))
            pt_on_edge = numpy.abs(distance_from_center - radius_per_pt)<epsilon
            count_pt_on_edge = scipy.ndimage.sum(pt_on_edge,
                                                 index,
                                                 range(s**2))
            count_pt_on_edge = numpy.array(count_pt_on_edge)
            #
            # Every dodecagon must have at least 2 points on the edge.
            #
            self.assertTrue(numpy.all(count_pt_on_edge>=2))
            #
            # Count the multi_edge cases
            #
            n_multi_edge += numpy.sum(count_pt_on_edge>=3)

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
        centers,eccentricity,major_axis_length,minor_axis_length =\
            morph.ellipse_from_second_moments(numpy.zeros((10,10)),
                                              numpy.zeros((10,10),int),
                                              [])
        self.assertEqual(centers.shape,(0,2))
        self.assertEqual(eccentricity.shape[0],0)
        self.assertEqual(major_axis_length.shape[0],0)
        self.assertEqual(minor_axis_length.shape[0],0)
    
    def test_00_01_zeros(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(numpy.zeros((10,10)),
                                              numpy.zeros((10,10),int),
                                              [1])
        self.assertEqual(centers.shape,(1,2))
        self.assertEqual(eccentricity.shape[0],1)
        self.assertEqual(major_axis_length.shape[0],1)
        self.assertEqual(minor_axis_length.shape[0],1)
    
    def test_01_01_rectangle(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(numpy.ones((10,20)),
                                              numpy.ones((10,20),int),
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
        img = numpy.zeros((101,101),int)
        y,x = numpy.mgrid[-50:51,-50:51]
        img[x*x+y*y<=2500] = 1
        centers,eccentricity,major_axis_length, minor_axis_length,theta =\
            morph.ellipse_from_second_moments(numpy.ones((101,101)),img,[1])
        self.assertAlmostEqual(eccentricity[0],0)
        self.assertWithinFraction(major_axis_length[0],100,.001)
        self.assertWithinFraction(minor_axis_length[0],100,.001)

class TestCalculateExtents(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure calculate_extents doesn't throw an exception if no image"""
        extents = morph.calculate_extents(numpy.zeros((10,10),int), [1])
    
    def test_01_01_square(self):
        """A square should have an extent of 1"""
        labels = numpy.zeros((10,10),int)
        labels[1:8,2:9]=1
        extents = morph.calculate_extents(labels,[1])
        self.assertAlmostEqual(extents,1)
    
    def test_01_02_circle(self):
        """A circle should have an extent of pi/4"""
        labels = numpy.zeros((1001,1001),int)
        y,x = numpy.mgrid[-500:501,-500:501]
        labels[x*x+y*y<=250000] = 1
        extents = morph.calculate_extents(labels,[1])
        self.assertAlmostEqual(extents,numpy.pi/4,2)

class TestCalculatePerimeters(unittest.TestCase):
    def test_00_00_zeros(self):
        """The perimeters of a zeros matrix should be all zero"""
        perimeters = morph.calculate_perimeters(numpy.zeros((10,10),int),[1])
        self.assertEqual(perimeters,0)
    
    def test_01_01_square(self):
        """The perimeter of a square should be the sum of the sides"""
        
        labels = numpy.zeros((10,10),int)
        labels[1:9,1:9] = 1
        perimeter = morph.calculate_perimeters(labels, [1])
        self.assertEqual(perimeter, 4*8)
        
    def test_01_02_circle(self):
        """The perimeter of a circle should be pi * diameter"""
        labels = numpy.zeros((101,101),int)
        y,x = numpy.mgrid[-50:51,-50:51]
        labels[x*x+y*y<=2500] = 1
        perimeter = morph.calculate_perimeters(labels, [1])
        epsilon = 20
        self.assertTrue(perimeter-numpy.pi*101<epsilon)

class TestCalculateConvexArea(unittest.TestCase):
    def test_00_00_degenerate_zero(self):
        """The convex area of an empty labels matrix should be zero"""
        labels = numpy.zeros((10,10),int)
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],0)
    
    def test_00_01_degenerate_point(self):
        """The convex area of a point should be 1"""
        labels = numpy.zeros((10,10),int)
        labels[4,4] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],1)

    def test_00_02_degenerate_line(self):
        """The convex area of a line should be its length"""
        labels = numpy.zeros((10,10),int)
        labels[1:9,4] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],8)
    
    def test_01_01_square(self):
        """The convex area of a square should be its area"""
        labels = numpy.zeros((10,10),int)
        labels[1:9,1:9] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertAlmostEqual(result[0],64)
    
    def test_01_02_cross(self):
        """The convex area of a cross should be the area of the enclosing diamond
        
        The area of a diamond is 1/2 of the area of the enclosing bounding box
        """
        labels = numpy.zeros((10,10),int)
        labels[1:9,4] = 1
        labels[4,1:9] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertAlmostEqual(result[0],32)
    
    def test_02_01_degenerate_point_and_line(self):
        """Test a degenerate point and line in the same image, out of order"""
        labels = numpy.zeros((10,10),int)
        labels[1,1] = 1
        labels[1:9,4] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertEqual(result[0],8)
        self.assertEqual(result[1],1)
    
    def test_02_02_degenerate_point_and_square(self):
        """Test a degenerate point and a square in the same image"""
        labels = numpy.zeros((10,10),int)
        labels[1,1] = 1
        labels[3:8,4:9] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertEqual(result[1],1)
        self.assertAlmostEqual(result[0],25)
    
    def test_02_03_square_and_cross(self):
        """Test two non-degenerate figures"""
        labels = numpy.zeros((20,10),int)
        labels[1:9,1:9] = 1
        labels[11:19,4] = 2
        labels[14,1:9] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertAlmostEqual(result[0],32)
        self.assertAlmostEqual(result[1],64)

class TestEulerNumber(unittest.TestCase):
    def test_00_00_even_zeros(self):
        labels = numpy.zeros((10,12),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_00_01_odd_zeros(self):
        labels = numpy.zeros((11,13),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_01_00_square(self):
        labels = numpy.zeros((10,12),int)
        labels[1:9,1:9] = 1
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],1)
        
    def test_01_01_square_with_hole(self):
        labels = numpy.zeros((10,12),int)
        labels[1:9,1:9] = 1
        labels[3:6,3:6] = 0
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_01_02_square_with_two_holes(self):
        labels = numpy.zeros((10,12),int)
        labels[1:9,1:9] = 1
        labels[2:4,2:8] = 0
        labels[6:8,2:8] = 0
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],-1)
    
    def test_02_01_square_touches_border(self):
        labels = numpy.ones((10,10),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],1)
        self.assertEqual(result[0],1)
        