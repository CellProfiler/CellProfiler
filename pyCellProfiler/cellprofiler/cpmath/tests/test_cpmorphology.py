"""test_cpmorphology - test the functions in cellprofiler.cpmath.cpmorphology

"""
__version__="$Revision$"

import unittest
import numpy
import scipy.ndimage

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
