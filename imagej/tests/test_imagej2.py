'''test_imagej2 - test the imagej2 module

'''
#
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2012 Broad Institute
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org

import numpy as np
import unittest

import bioformats
import cellprofiler.utilities.jutil as J
import imagej.imagej2 as ij2

class TestImagej2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = ij2.create_context(None)
        
    @classmethod
    def tearDownClass(cls):
        del cls.context
        J.static_call('java/lang/System', 'gc', '()V')
        
    def tearDown(self):
        # Close all displays after each test
        svc = ij2.get_display_service(self.context)
        for display in svc.getDisplays():
            display.close()
        
    def test_01_01_get_service(self):
        self.assertIsNotNone(
            self.context.getService("imagej.data.DatasetService"))
        
    def test_02_01_get_module_service(self):
        self.assertIsNotNone(ij2.get_module_service(self.context))
        
    def test_02_02_get_modules(self):
        svc = ij2.get_module_service(self.context)
        module_infos = svc.getModules()
        self.assertTrue(J.is_instance_of(module_infos[0].o,
                                         "imagej/module/ModuleInfo"))
        
    def test_02_03_module_info(self):
        svc = ij2.get_module_service(self.context)
        #
        # Run the methods without checks on the values to check signatures
        module_infos = svc.getModules()
        module_info = module_infos[0]
        module_info.getMenuRoot()
        module_info.getMenuPath()
        module_info.createModule()
        module_info.getTitle()
        module_info.getName()
        module_info.getClassName()
        for module_info in module_infos:
            inputs = module_info.getInputs()
            if len(inputs) > 0:
                module_item = inputs[0]
                self.assertIsNotNone(module_info.getInput(module_item.getName()))
                break
        else:
            raise AssertionError("No input items found")
        for module_info in module_infos:
            outputs = module_info.getOutputs()
            if len(outputs) > 0:
                module_item = outputs[0]
                self.assertIsNotNone(module_info.getOutput(module_item.getName()))
                break
        else:
            raise AssertionError("No output items found")
        
    def test_02_04_module_item(self):
        svc = ij2.get_module_service(self.context)
        module_infos = svc.getModules()
        for module_info in module_infos:
            inputs = module_info.getInputs()
            if len(inputs) > 0:
                module_item = inputs[0]
                module_item.getType()
                module_item.getWidgetStyle()
                module_item.getMinimumValue()
                module_item.getMaximumValue()
                module_item.getStepSize()
                module_item.getName()
                module_item.getLabel()
                module_item.getDescription()
                module_item.loadValue()
                self.assertTrue(module_item.isInput())
                module_item.isOutput()
                break
        
    def test_02_05_module(self):
        svc = ij2.get_module_service(self.context)
        module_infos = svc.getModules()
        for module_info in module_infos:
            if module_info.getClassName() == \
               'imagej.core.commands.assign.AddSpecifiedNoiseToDataValues':
                module = module_info.createModule()
                module.getInfo()
                module.getInput('stdDev')
                module.getOutput('display')
                module.setInput('stdDev', 2.5)
                module.setOutput('display', None)
                module.isResolved('display')
                module.setResolved('display', False)
                break
        else:
            raise AssertionError("Could not find target module")
                
    def test_02_06_menu_entry(self):
        svc = ij2.get_module_service(self.context)
        module_infos = svc.getModules()
        for module_info in module_infos:
            if module_info.getClassName() == \
               'imagej.core.commands.assign.AddSpecifiedNoiseToDataValues':
                menu_path = module_info.getMenuPath()
                for item in J.iterate_collection(menu_path):
                    menu_entry = ij2.wrap_menu_entry(item)
                    menu_entry.getName()
                    menu_entry.setName("Foo")
                    menu_entry.getWeight()
                    menu_entry.setWeight(5)
                    menu_entry.getMnemonic()
                    menu_entry.setMnemonic("X")
                    menu_entry.getAccelerator()
                    menu_entry.setAccelerator(None)
                    menu_entry.getIconPath()
                    menu_entry.setIconPath(None)
                break
        else:
            raise AssertionError("Could not find target module")
        
    def test_03_01_command_service(self):
        svc = ij2.get_command_service(self.context)
        
    def test_03_02_command_service_run(self):
        svc = ij2.get_command_service(self.context)
        module_infos = ij2.get_module_service(self.context).getModules()
        for module_info in module_infos:
            if module_info.getClassName() == \
               'imagej.core.commands.app.AboutImageJ':
                d = J.get_map_wrapper(J.make_instance('java/util/HashMap', '()V'))
                d['context'] = self.context
                future = svc.run(module_info.o, d.o)
                module = J.call(future, "get", "()Ljava/lang/Object;")
                module = ij2.wrap_module(module)
                module.getOutput('display')
                break
        else:
            raise AssertionError("Could not find target module")
        
    def test_04_01_axes(self):
        ij2.Axes().CHANNEL
        ij2.Axes().X
        ij2.Axes().Y
        
    def test_05_01_dataset_service(self):
        svc = ij2.get_dataset_service(self.context)
        
    def test_05_02_create1(self):
        svc = ij2.get_dataset_service(self.context)
        result = svc.create1(np.array([10, 15]), 
                             "MyDataset", [ij2.Axes().X, ij2.Axes().Y],
                             8, False, False)
        self.assertEqual(result.getName(), "MyDataset")
        self.assertTrue(result.isInteger())
        self.assertFalse(result.isSigned())
        result.getImgPlus()
        result.getType()
        result.calibration(0)
        
    def test_05_03_get_pixel_data(self):
        svc = ij2.get_dataset_service(self.context)
        result = svc.create1(np.array([10, 15]), 
                             "MyDataset", [ij2.Axes().Y, ij2.Axes().X],
                             8, False, False)
        imgplus = result.getImgPlus()
        pixel_data = ij2.get_pixel_data(imgplus)
        self.assertSequenceEqual(pixel_data.shape, [10, 15])
        
    def test_05_04_get_dataset_pixel_data(self):
        # test ij2.get_pixel_data
        svc = ij2.get_dataset_service(self.context)
        result = svc.create1(np.array([10, 15]), 
                             "MyDataset", [ij2.Axes().Y, ij2.Axes().X],
                             8, False, False)
        pixel_data = result.get_pixel_data()
        self.assertSequenceEqual(pixel_data.shape, [10, 15])
        
    def test_05_05_get_dataset_pixel_data_axes(self):
        svc = ij2.get_dataset_service(self.context)
        result = svc.create1(np.array([10, 15]), 
                             "MyDataset", [ij2.Axes().Y, ij2.Axes().X],
                             8, False, False)
        pixel_data = result.get_pixel_data(axes = [ij2.Axes().X, ij2.Axes().Y])
        self.assertSequenceEqual(pixel_data.shape, [15, 10])
        
    def test_05_06_create_dataset(self):
        r = np.random.RandomState()
        r.seed(56)
        image = r.randint(0, 256, (11,13))
        ds = ij2.create_dataset(self.context, image, "Foo")
        pixel_data = ds.get_pixel_data()
        np.testing.assert_array_equal(image, pixel_data)
        
    def test_07_01_wrap_interval(self):
        svc = ij2.get_dataset_service(self.context)
        result = svc.create1(np.array([10, 15]), 
                             "MyDataset", [ij2.Axes().Y, ij2.Axes().X],
                             8, False, False)
        imgplus = ij2.wrap_interval(result.getImgPlus())
        self.assertEqual(imgplus.min1D(0), 0)
        self.assertEqual(imgplus.max1D(1), 14)
        self.assertSequenceEqual(imgplus.minND(), [0, 0])
        self.assertSequenceEqual(imgplus.maxND(), [9, 14])
        self.assertSequenceEqual(imgplus.dimensions(), [10, 15])
        
    def test_08_01_create_overlay(self):
        r = np.random.RandomState()
        r.seed(81)
        mask = r.uniform(size=(23,10)) > .5
        overlay = ij2.create_overlay(self.context, mask)
        roi = J.call(overlay, "getRegionOfInterest",
                     "()Lnet/imglib2/roi/RegionOfInterest;")
        i = r.randint(0, 23, 10)
        j = r.randint(0, 10, 10)
        for ii, jj in zip(i, j):
            location = np.array([jj, ii], float)
            test = J.call(roi, "contains", "([D)Z", location)
            self.assertEqual(mask[ii, jj], test)
    
    def test_09_01_get_display_service(self):
        svc = ij2.get_display_service(self.context)
        
    def test_09_02_create_display(self):
        svc = ij2.get_display_service(self.context)
        r = np.random.RandomState()
        r.seed(92)
        image = r.randint(0, 256, (11,13))
        ds = ij2.create_dataset(self.context, image, "Foo")
        display = svc.createDisplay("foo", ds)
        self.assertEqual(display.size(), 1)
        self.assertFalse(display.isEmpty())
        self.assertFalse(display.contains(None))
        views = display.toArray()
        self.assertEqual(len(views), 1)

        self.assertTrue(display.canDisplay(ds.o))
        display.update()
        self.assertEqual(display.getName(), "Foo")
        display.setName("Bar")
        self.assertEqual(display.getName(), "Bar")
        display.getActiveView()
        display.getActiveAxis()
        display.getCanvas()
        display.setActiveAxis(ij2.Axes().X)
        
    def test_09_03_set_active_display(self):
        svc = ij2.get_display_service(self.context)
        r = np.random.RandomState()
        r.seed(92)
        image = r.randint(0, 256, (11,13))
        ds = ij2.create_dataset(self.context, image, "Foo")
        display = svc.createDisplay("foo", ds)
        svc.setActiveDisplay(display.o)
        
    def test_09_04_get_active_display(self):
        svc = ij2.get_display_service(self.context)
        r = np.random.RandomState()
        r.seed(92)
        image = r.randint(0, 256, (11,13))
        ds = ij2.create_dataset(self.context, image, "Foo")
        display = svc.createDisplay("foo", ds)
        ds = ij2.create_dataset(self.context, image, "Bar")
        svc.createDisplay("bar", ds)
        svc.setActiveDisplay(display.o)
        self.assertEqual(svc.getActiveDisplay().getName(), "Foo")
        
    def test_09_05_get_active_image_display(self):
        svc = ij2.get_display_service(self.context)
        r = np.random.RandomState()
        r.seed(92)
        image = r.randint(0, 256, (11,13))
        ds = ij2.create_dataset(self.context, image, "Foo")
        svc.createDisplay("foo", ds)
        ds = ij2.create_dataset(self.context, image, "Bar")
        display = svc.createDisplay("bar", ds)
        svc.setActiveDisplay(display.o)
        self.assertEqual(svc.getActiveImageDisplay().getName(), "Bar")
        
    def test_09_06_get_display_by_name(self):
        svc = ij2.get_display_service(self.context)
        r = np.random.RandomState()
        r.seed(92)
        image = r.randint(0, 256, (11,13))
        ds = ij2.create_dataset(self.context, image, "Foo")
        svc.createDisplay("foo", ds)
        image = r.randint(0, 256, (14,12))
        ds2 = ij2.create_dataset(self.context, image, "Bar")
        display = svc.createDisplay("bar", ds)
        svc.setActiveDisplay(display.o)
        display = svc.getDisplay("Foo")
        view = display.getActiveView()
        ds3 = ij2.wrap_interval(view.getData())
        self.assertSequenceEqual(ds3.dimensions(), [13, 11])
        
    def test_09_07_is_unique_name(self):
        svc = ij2.get_display_service(self.context)
        r = np.random.RandomState()
        r.seed(92)
        image = r.randint(0, 256, (11,13))
        ds = ij2.create_dataset(self.context, image, "Foo")
        svc.createDisplay("foo", ds)
        self.assertTrue(svc.isUniqueName("Bar"))
        self.assertFalse(svc.isUniqueName("Foo"))
        
    def test_10_01_get_script_service(self):
        svc = ij2.get_script_service(self.context)
        svc.getPluginService()
        svc.getLogService()
        svc.getIndex()
        for factory in svc.getLanguages():
            factory.getLanguageName()
        
    def test_10_02_get_by_name(self):
        svc = ij2.get_script_service(self.context)
        factory = svc.getByName("ECMAScript")
        self.assertIsNotNone(factory)
        
    def test_10_03_get_by_file_extension(self):
        svc = ij2.get_script_service(self.context)
        self.assertIsNotNone(svc.getByFileExtension("foo.js"))
        
    def test_10_04_engine_factory_wrapper(self):
        svc = ij2.get_script_service(self.context)
        factory = svc.getByName("ECMAScript")
        factory.getEngineName()
        factory.getEngineVersion()
        factory.getExtensions()
        factory.getMimeTypes()
        factory.getNames()
        factory.getLanguageName()
        factory.getLanguageVersion()
        factory.getParameter(J.get_static_field('javax/script/ScriptEngine',
                                                'NAME', 'Ljava/lang/String;'))
        factory.getMethodCallSyntax("myobject", "mymethod", ["param1", "param2"])
        factory.getOutputStatement("Hello, world")
        factory.getProgram(["I.do.this()", "I.do.that()"])
        factory.getScriptEngine()
        
    def test_10_05_script_engine(self):
        svc = ij2.get_script_service(self.context)
        factory = svc.getByName("ECMAScript")
        engine = factory.getScriptEngine()
        engine.ARGV
        engine.ENGINE
        engine.FILENAME
        engine.ENGINE_VERSION
        engine.NAME
        engine.LANGUAGE
        engine.LANGUAGE_VERSION
        engine.ENGINE_SCOPE
        engine.GLOBAL_SCOPE
        engine.createBindings()
        engine.setBindings(engine.getBindings(engine.ENGINE_SCOPE), 
                           engine.ENGINE_SCOPE)
        engine.setContext(engine.getContext())
        engine.getFactory()
        
    def test_10_06_get_put(self):
        svc = ij2.get_script_service(self.context)
        factory = svc.getByName("ECMAScript")
        engine = factory.getScriptEngine()
        engine.put("Foo", "Bar")
        self.assertEqual(engine.get("Foo"), "Bar")
        
    def test_10_07_evalS(self):
        svc = ij2.get_script_service(self.context)
        factory = svc.getByName("ECMAScript")
        engine = factory.getScriptEngine()
        result = engine.evalS("2+2")
        self.assertEqual(J.call(result, "intValue", "()I"), 4)
        
    def test_10_08_eval_with_bindings(self):
        svc = ij2.get_script_service(self.context)
        factory = svc.getByName("ECMAScript")
        engine = factory.getScriptEngine()
        engine.put("a", 2)
        engine.evalS("var b = a+a;")
        self.assertEqual(J.call(engine.get("b"), "intValue", "()I"), 4)