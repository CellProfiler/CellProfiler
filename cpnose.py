"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

from unittest.case import SkipTest
from pkg_resources import load_entry_point
import logging
import logging.handlers
logger = logging.getLogger(__name__)
import javabridge
import nose
import nose.plugins
import os
import tempfile
import sys
from xml.dom.minidom import parse
from nose.suite import ContextSuiteFactory
from nose.config import Config
from nose.util import resolve_name
from nose.plugins.manager import PluginManager

from cellprofiler.utilities.cpjvm import \
     get_path_to_jars, get_jars, get_patcher_args

import numpy as np
np.seterr(all='ignore')

class CPShutdownPlugin(nose.plugins.Plugin):
    '''CellProfiler shutdown plugin
    
    This plugin shuts down threads that might happen to have been
    left open.
    '''
    name = "cpshutdown"
    score = 101
    enabled = False
    
    def begin(self):
        #
        # We monkey-patch javabridge.start_vm here in order to
        # set up the ImageJ event bus (actually 
        # org.bushe.swing.event.ThreadSafeEventService) to not start
        # its cleanup thread which semi-buggy hangs around forever
        # and prevents Java from exiting.
        #
        def patch_start_vm(*args, **kwargs):
            jvm_args = list(args[0]) +  [
                "-Dloci.bioformats.loaded=true",
                "-Djava.util.prefs.PreferencesFactory="+
                "org.cellprofiler.headlesspreferences"+
                ".HeadlessPreferencesFactory"]
            #
            # Find the ij1patcher
            #
            if hasattr(sys, 'frozen') and sys.platform == 'win32':
                root = os.path.dirname(sys.argv[0])
            else:
                root = os.path.dirname(__file__)
            jardir = os.path.join(root, "imagej", "jars")
            patchers = sorted([
                    x for x in os.listdir(jardir)
                    if x.startswith("ij1-patcher") and x.endswith(".jar")])
            if len(patchers) > 0:
                jvm_args.append(
                    "-javaagent:%s=init" % os.path.join(jardir, patchers[-1]))
            result = start_vm(jvm_args, *args[1:], **kwargs)
            if javabridge.get_env() is not None:
                try:
                    event_service_cls = javabridge.JClassWrapper(
                        "org.bushe.swing.event.ThreadSafeEventService")
                    event_service_cls.CLEANUP_PERIOD_MS_DEFAULT = None
                except:
                    pass
            return result
        patch_start_vm.func_globals["start_vm"] = javabridge.start_vm
        javabridge.start_vm = patch_start_vm
        if "CP_EXAMPLEIMAGES" in os.environ:
            self.temp_exampleimages = None
        else:
            self.temp_exampleimages = tempfile.mkdtemp(prefix="cpexampleimages")

        if "CP_TEMPIMAGES" in os.environ:
            self.temp_images = None
        else:
            self.temp_images = tempfile.mkdtemp(prefix="cptempimages")
        #
        # Set up matplotlib for WXAgg if in frozen mode
        # otherwise it looks for TK which isn't there
        #
        if hasattr(sys, 'frozen'):
            import matplotlib
            matplotlib.use("WXAgg")
        try:
            from ilastik.core.jobMachine import GLOBAL_WM
            GLOBAL_WM.set_thread_count(1)
        except:
            pass
            
    def finalize(self, result):
        try:
            if self.temp_images is not None:
                import shutil
                shutil.rmtree(self.temp_images)
            if self.temp_exampleimages is not None:
                import shutil
                shutil.rmtree(self.temp_exampleimages)
        except:
            pass
        from cellprofiler.utilities.cpjvm import cp_stop_vm
        try:
            import wx
            app = wx.GetApp()
            if app is not None:
                app.ExitMainLoop()
                app.MainLoop()
            cp_stop_vm(kill=False)
        except:
            logging.root.warn("Failed to shut down AWT", exc_info=1)
        try:
            from ilastik.core.jobMachine import GLOBAL_WM
            GLOBAL_WM.stopWorkers()
        except:
            logging.root.warn("Failed to stop Ilastik")
        try:
            from cellprofiler.utilities.zmqrequest import join_to_the_boundary
            join_to_the_boundary()
        except:
            logging.root.warn("Failed to stop zmq boundary")

    def wantFile(self, filename):
        if filename.endswith("setup.py"):
            return False
        
    def wantDirectory(self, dirname):
        if dirname.endswith("tutorial"):
            return False
        
class LoadTestsFromXML(nose.plugins.Plugin):
    '''Load tests from an XML file generated by XUNIT'''
    name = "load-tests-from-xml"
    enabled = False
    
    def options(self, parser, env):
        parser.add_option(
            "--xml-test-file",
            action="store",
            default=env.get('NOSE_XML_TEST_FILE'),
            metavar="XML_TEST_FILE",
            dest="xml_test_file",
            help="Load tests to run from an XML generated by XUNIT")
        
    def configure(self, options, conf):
        super(LoadTestsFromXML, self).configure(options, conf)
        if options.xml_test_file:
            self.enabled = True
            self.xml_test_file = options.xml_test_file
            self.config = conf
            
    def prepareTestLoader(self, loader):
        '''Replace the loader with ourself'''
        if self.enabled:
            return self
    
    def loadTestsFromNames(self, names, module=None):
        '''Instead of processing the names, return the tests in the XML file'''
        if self.enabled:
            tests = get_tests_from_xml_files([self.xml_test_file], self.config)
            return tests
        
        
class MockModule(object):
    def __getattr__(self, *args, **kwargs):
        raise SkipTest

    def __getitem__(self, *args, **kwargs):
        raise SkipTest
    
def get_tests_from_xml_files(paths, config):
    '''Retrieve the suite of tests from an xml file produced by the xunit plugin
    
    Start cpnose like this to make the file:
    
    python cpnose.py --collect-only --with-xunit --xunit-file=<path>
    
    paths - paths to xml files
    
    returns a test suite with the paths
    '''
    factory = ContextSuiteFactory(config=config)
    hierarchy = {}
    classes = {}
    for path in paths:
        xml = parse(path)
        for test_suite in xml.getElementsByTagName("testsuite"):
            for test_case in test_suite.getElementsByTagName("testcase"):
                full_class_name = test_case.getAttribute("classname")
                name = test_case.getAttribute("name")
                parts = full_class_name.split(".")
                leaf = hierarchy
                for part in parts:
                    if part not in leaf:
                        leaf[part] = {}
                    leaf = leaf[part]
                module_name, class_name = full_class_name.rsplit(".", 1)
                try:
                    if full_class_name not in classes:
                        klass = resolve_name(full_class_name)
                        classes[full_class_name] = klass
                    else:
                        klass = classes[full_class_name]
                    if klass is None:
                        continue
                    test = klass(name)
                    leaf[name] = test
                except:
                    logger.warning("Failed to load test %s.%s" %
                                   (full_class_name, name), exc_info=True)
    return get_suite_from_dictionary(factory, hierarchy)

def get_suite_from_dictionary(factory, d, parts = []):
    '''Recursively combine the values in a dictionary into a suite'''
    tests = []
    for key in sorted(d.keys()):
        test = d[key]
        if isinstance(test, dict):
            test = get_suite_from_dictionary(factory, test, parts + [key])
        tests.append(test)
    suite = factory(tests)
    return suite
    
def main(*args):
    '''Run the CellProfiler nose tests'''
    args = list(args)
    import cellprofiler.preferences as cpprefs
    cpprefs.set_headless()
    cpprefs.set_awt_headless(True)
    if '--noguitests' in args:
        args.remove('--noguitests')
        sys.modules['wx'] = MockModule()
        sys.modules['wx.html'] = MockModule()
        import matplotlib
        matplotlib.use('agg')
        with_guitests = False
        wxapp = None
    else:
        with_guitests = True
        import wx
        wxapp = wx.App(False)

    def mock_start_vm(*args, **kwargs):
        raise SkipTest
    
    if '--nojavatests' in args:
        args.remove('--nojavatests')
        javabridge.start_vm = mock_start_vm

    addplugins = [CPShutdownPlugin()]

    if not "--with-cpshutdown" in args:
        args.append("--with-cpshutdown")
    if '--with-kill-vm' in args:
        args[args.index('--with-kill-vm')] = '--with-javabridge'
    if '--with-javabridge' in args:
        class_path = get_jars()
        args.append('--classpath=%s' % os.pathsep.join(class_path))
        if hasattr(sys, "frozen"):
            from javabridge.noseplugin import JavabridgePlugin
            javabridge_plugin_class = JavabridgePlugin
            addplugins.append(JavabridgePlugin())
        else:
            javabridge_plugin_class = load_entry_point(
                'javabridge', 'nose.plugins.0.10', 'javabridge')
        javabridge_plugin_class.extra_jvm_args += get_patcher_args(class_path)
        if "CP_JDWP_PORT" in os.environ:
            javabridge_plugin_class.extra_jvm_args.append(
                ("-agentlib:jdwp=transport=dt_socket,address=127.0.0.1:%s"
                 ",server=y,suspend=n") % os.environ["CP_JDWP_PORT"])
        addplugins.append(javabridge_plugin_class())
        #
        # Run the shutdown plugin before the javabridge plugin when exiting
        #
        CPShutdownPlugin.score = javabridge_plugin_class.score + 1
        
    addplugins.append(LoadTestsFromXML())

    if len(args) == 0:
        args = ['--testmatch=(?:^)test_.*']

    args += ['--exe']
    if hasattr(sys, "frozen"):
        #
        # Patch nose.core.TestProgram.usage
        #    it will attempt to get usage.txt and it's not worth
        #    the effort to make that work.
        #
        from nose.core import TestProgram
        TestProgram.usage = lambda cls: ""
    
    nose.main(argv=args, addplugins=addplugins)
        
if __name__ == "__main__":
    main(*sys.argv)
    print "At bottom of main"
    os._exit(0)
