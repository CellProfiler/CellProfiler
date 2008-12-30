"""test_variable.py - test cellprofiler.variable

"""
__version__="$Revision: 1$"

import unittest
import cellprofiler.variable as vvv

class TestVariable(unittest.TestCase):
    def test_00_00_init(self):
        x=vvv.Variable("text","value")
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,"value")
        self.assertTrue(x.key())
    
    def test_01_01_equality(self):
        x=vvv.Variable("text","value")
        self.assertTrue(x == "value")
        self.assertTrue(x != "text")
        self.assertFalse(x != "value")
        self.assertFalse(x == "text")
        self.assertEqual(x.value,"value")
    
class TestText(unittest.TestCase):
    def test_00_00_init(self):
        x=vvv.Text("text","value")
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,"value")
        self.assertTrue(x.key())

class TestInteger(unittest.TestCase):
    def test_00_00_init(self):
        x=vvv.Integer("text",5)
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,5)
    
    def test_01_01_numeric_value(self):
        x=vvv.Integer("text",5)
        self.assertTrue(isinstance(x.value,int))
    
    def test_01_02_equality(self):
        x=vvv.Integer("text",5)
        self.assertTrue(x==5)
        self.assertTrue(x.value==5)
        self.assertFalse(x==6)
        self.assertTrue(x!=6)

    def test_01_03_assign_str(self):
        x=vvv.Integer("text",5)
        x.value = 6
        self.assertTrue(x==6)
    
    def test_02_01_neg_assign_number(self):
        x=vvv.Integer("text",5)
        self.assertRaises(ValueError, x.set_value,"foo")

class TestBinary(unittest.TestCase):
    def test_00_01_init_true(self):
        x=vvv.Binary("text",True)
        self.assertEqual(x.text,"text")
        self.assertTrue(x.value==True)
        self.assertTrue(x == True)
        self.assertTrue(x != False)
        self.assertFalse(x != True)
        self.assertFalse(x == False)
        
    def test_00_02_init_false(self):
        x=vvv.Binary("text",False)
        self.assertTrue(x.value==False)
        self.assertFalse(x == True)
        self.assertFalse(x != False)
        self.assertTrue(x != True)
    
    def test_01_01_set_true(self):
        x=vvv.Binary("text",False)
        x.value = True
        self.assertTrue(x.value==True)
    
    def test_01_02_set_false(self):
        x=vvv.Binary("text",True)
        x.value = False
        self.assertTrue(x.value==False)
    
class TestChoice(unittest.TestCase):
    def test_00_00_init(self):
        x=vvv.Choice("text",["choice"])
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,"choice")
        self.assertEqual(len(x.choices),1)
        self.assertEqual(x.choices[0],"choice")
    
    def test_01_01_assign(self):
        x=vvv.Choice("text",["foo","bar"],"bar")
        self.assertTrue(x == "bar")
        x.value = "foo"
        self.assertTrue(x == "foo")
        x.value = "bar"
        self.assertTrue(x == "bar")
        
    def test_02_01_neg_assign(self):
        x=vvv.Choice("text",["choice"])
        self.assertRaises(ValueError, x.set_value,"foo")

class TestCustomChoice(unittest.TestCase):
    def test_00_00_init(self):
        x=vvv.CustomChoice("text",["choice"])
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,"choice")
        self.assertEqual(len(x.choices),1)
        self.assertEqual(x.choices[0],"choice")
    
    def test_01_01_assign(self):
        x=vvv.CustomChoice("text",["foo","bar"],"bar")
        self.assertTrue(x == "bar")
        x.value = "foo"
        self.assertTrue(x == "foo")
        x.value = "bar"
        self.assertTrue(x == "bar")
        
    def test_01_02_assign_other(self):
        x=vvv.CustomChoice("text",["foo","bar"],"bar")
        x.value = "other"
        self.assertTrue(x == "other")
        self.assertEqual(len(x.choices),3)
        self.assertEqual(x.choices[0],"other")

    