"""test_variable.py - test cellprofiler.settings

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision: 1$"

import unittest
import cellprofiler.settings as cps

class TestVariable(unittest.TestCase):
    def test_00_00_init(self):
        x=cps.Setting("text","value")
        x.test_valid(None)
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,"value")
        self.assertTrue(x.key())
    
    def test_01_01_equality(self):
        x=cps.Setting("text","value")
        self.assertTrue(x == "value")
        self.assertTrue(x != "text")
        self.assertFalse(x != "value")
        self.assertFalse(x == "text")
        self.assertEqual(x.value,"value")
    
class TestText(unittest.TestCase):
    def test_00_00_init(self):
        x=cps.Text("text","value")
        x.test_valid(None)
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,"value")
        self.assertTrue(x.key())

class TestInteger(unittest.TestCase):
    def test_00_00_init(self):
        x=cps.Integer("text",5)
        x.test_valid(None)
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,5)
    
    def test_01_01_numeric_value(self):
        x=cps.Integer("text",5)
        self.assertTrue(isinstance(x.value,int))
    
    def test_01_02_equality(self):
        x=cps.Integer("text",5)
        self.assertTrue(x==5)
        self.assertTrue(x.value==5)
        self.assertFalse(x==6)
        self.assertTrue(x!=6)

    def test_01_03_assign_str(self):
        x=cps.Integer("text",5)
        x.value = 6
        self.assertTrue(x==6)
    
    def test_02_01_neg_assign_number(self):
        x=cps.Integer("text",5)
        x.set_value("foo")
        self.assertRaises(ValueError, x.test_valid, None)

class TestBinary(unittest.TestCase):
    def test_00_01_init_true(self):
        x=cps.Binary("text",True)
        x.test_valid(None)
        self.assertEqual(x.text,"text")
        self.assertTrue(x.value==True)
        self.assertTrue(x == True)
        self.assertTrue(x != False)
        self.assertFalse(x != True)
        self.assertFalse(x == False)
        
    def test_00_02_init_false(self):
        x=cps.Binary("text",False)
        self.assertTrue(x.value==False)
        self.assertFalse(x == True)
        self.assertFalse(x != False)
        self.assertTrue(x != True)
    
    def test_01_01_set_true(self):
        x=cps.Binary("text",False)
        x.value = True
        self.assertTrue(x.value==True)
    
    def test_01_02_set_false(self):
        x=cps.Binary("text",True)
        x.value = False
        self.assertTrue(x.value==False)
        
class TestChoice(unittest.TestCase):
    def test_00_00_init(self):
        x=cps.Choice("text",["choice"])
        x.test_valid(None)
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,"choice")
        self.assertEqual(len(x.choices),1)
        self.assertEqual(x.choices[0],"choice")
    
    def test_01_01_assign(self):
        x=cps.Choice("text",["foo","bar"],"bar")
        self.assertTrue(x == "bar")
        x.value = "foo"
        self.assertTrue(x == "foo")
        x.value = "bar"
        self.assertTrue(x == "bar")
        
    def test_02_01_neg_assign(self):
        x=cps.Choice("text",["choice"])
        x.set_value("foo")
        self.assertRaises(ValueError, x.test_valid, None)

class TestCustomChoice(unittest.TestCase):
    def test_00_00_init(self):
        x=cps.CustomChoice("text",["choice"])
        x.test_valid(None)
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,"choice")
        self.assertEqual(len(x.choices),1)
        self.assertEqual(x.choices[0],"choice")
    
    def test_01_01_assign(self):
        x=cps.CustomChoice("text",["foo","bar"],"bar")
        self.assertTrue(x == "bar")
        x.value = "foo"
        self.assertTrue(x == "foo")
        x.value = "bar"
        self.assertTrue(x == "bar")
        
    def test_01_02_assign_other(self):
        x=cps.CustomChoice("text",["foo","bar"],"bar")
        x.value = "other"
        self.assertTrue(x == "other")
        self.assertEqual(len(x.choices),3)
        self.assertEqual(x.choices[0],"other")

class TestIntegerRange(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.IntegerRange("text",(1,2),1,5)
        x.test_valid(None)
        self.assertEqual(x.text,"text")
        self.assertEqual(str(x),"1,2")
        self.assertEqual(x.min,1)
        self.assertEqual(x.max,2)
        x.test_valid(None)
    
    def test_01_01_assign_tuple(self):
        x = cps.IntegerRange("text",(1,2),1,5)
        x.value = (2,5)
        self.assertEqual(x.min,2)
        self.assertEqual(x.max,5)
        x.test_valid(None)
    
    def test_01_02_assign_string(self):
        x = cps.IntegerRange("text",(1,2),1,5)
        x.value = "2,5"
        self.assertEqual(x.min,2)
        self.assertEqual(x.max,5)
        x.test_valid(None)

    def test_02_01_neg_min(self):
        x = cps.IntegerRange("text",(1,2),1,5)
        x.value = (0,2)
        self.assertRaises(ValueError,x.test_valid,None)

    def test_02_02_neg_max(self):
        x = cps.IntegerRange("text",(1,2),1,5)
        x.value = (1,6)
        self.assertRaises(ValueError,x.test_valid,None)
    
    def test_02_03_neg_order(self):
        x = cps.IntegerRange("text",(1,2),1,5)
        x.value = (2,1)
        self.assertRaises(ValueError,x.test_valid,None)
        
    def test_03_01_no_range(self):
        """Regression test a bug where the variable throws an exception if there is no range"""
        x=cps.IntegerRange("text",(1,2))
        x.test_valid(None)

class TestFloatRange(unittest.TestCase):
    def test_00_00_init(self):
        x = cps.FloatRange("text",(1,2),1,5)
        x.test_valid(None)
        self.assertEqual(x.text,"text")
        self.assertEqual(x.value,(1,2))
        self.assertEqual(x.min,1)
        self.assertEqual(x.max,2)
        x.test_valid(None)
    
    def test_01_01_assign_tuple(self):
        x = cps.FloatRange("text",(1,2),1,5)
        x.value = (2,5)
        self.assertEqual(x.min,2)
        self.assertEqual(x.max,5)
        x.test_valid(None)
    
    def test_01_02_assign_string(self):
        x = cps.FloatRange("text",(1,2),1,5)
        x.value = "2,5"
        self.assertEqual(x.min,2)
        self.assertEqual(x.max,5)
        x.test_valid(None)

    def test_02_01_neg_min(self):
        x = cps.FloatRange("text",(1,2),1,5)
        x.value = (0,2)
        self.assertRaises(ValueError,x.test_valid,None)

    def test_02_02_neg_max(self):
        x = cps.FloatRange("text",(1,2),1,5)
        x.value = (1,6)
        self.assertRaises(ValueError,x.test_valid,None)
    
    def test_02_03_neg_order(self):
        x = cps.FloatRange("text",(1,2),1,5)
        x.value = (2,1)
        self.assertRaises(ValueError,x.test_valid,None)

    def test_03_01_no_range(self):
        """Regression test a bug where the variable throws an exception if there is no range"""
        x=cps.FloatRange("text",(1,2))
        x.test_valid(None)
