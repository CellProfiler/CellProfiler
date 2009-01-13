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
            
