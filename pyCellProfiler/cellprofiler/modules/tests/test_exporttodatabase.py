'''test_exporttodatabase.py - test the ExportToDatabase module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import base64
import numpy as np
import os
import Image as PILImage
import scipy.ndimage
from StringIO import StringIO
import tempfile
import unittest
import uuid
import zlib

import cellprofiler.cpmodule as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.workspace as cpw
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpmeas

import cellprofiler.modules.exporttodatabase as E

M_CATEGORY = "my"
OBJ_FEATURE = 'objmeasurement'
INT_IMG_FEATURE = 'int_imagemeasurement'
FLOAT_IMG_FEATURE = 'float_imagemeasurement'
STRING_IMG_FEATURE = 'string_imagemeasurement'
OBJ_MEASUREMENT, INT_IMG_MEASUREMENT, FLOAT_IMG_MEASUREMENT,\
    STRING_IMG_MEASUREMENT = ['_'.join((M_CATEGORY, x))
                              for x in (OBJ_FEATURE, INT_IMG_FEATURE, 
                                        FLOAT_IMG_FEATURE, STRING_IMG_FEATURE)]
OBJECT_NAME = 'myobject'
IMAGE_NAME = 'myimage'
OBJECT_COUNT_MEASUREMENT = 'Count_%s'%OBJECT_NAME

INT_VALUE = 10
FLOAT_VALUE = 15.5
STRING_VALUE = "Hello, world"
OBJ_VALUE = np.array([1.5, 3.67, 2.8])

class TestExportToDatabase(unittest.TestCase):
    def setUp(self):
        self.__cursor = None
    
    @property
    def cursor(self):
        if self.__cursor is None:
            import MySQLdb
            from MySQLdb.cursors import SSCursor
            connection = MySQLdb.connect(host='imgdb01',
                                         user='cpuser',
                                         passwd='cPus3r')
            self.__cursor = SSCursor(connection)
        return self.__cursor
    
    def get_sqlite_cursor(self, module):
        from pysqlite2 import dbapi2 as sqlite
        self.assertTrue(isinstance( module, E.ExportToDatabase))
        file_name = os.path.join(module.output_directory.value,
                                 module.sqlite_file.value)
        connection = sqlite.connect(file_name)
        cursor = connection.cursor()
        return cursor, connection
    
    def test_01_01_load_matlab_10(self):
        data = ('eJwBNATL+01BVExBQiA1LjAgTUFULWZpbGUsIFBsYXRmb3JtOiBQQ1dJTiwg'
                'Q3JlYXRlZCBvbjogVHVlIE9jdCAxMyAwOTowNzoxMyAyMDA5ICAgICAgICAg'
                'ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAAAUlN'
                'DwAAAKwDAAB4nOxYbW/TMBB2t26CIU2FCWnfCN/4sJWEF4mPeymCStta1qqA'
                'ACG3cTuDG1eJM7Ug/tt+GvaWtK5JiZslWUCLFKXn+p7zc3e2z94EAFw8BmCd'
                'f+/wdwVcPWuBXJJeIbcQY9gZeGugDLaD9gv+dqCLYZegDiQ+8sD0CdvrTp+2'
                'J6PpX8fU9gk6gUO5M39O/GEXuV6jHyoGfzfxGJEW/oHA/BN2O0Xn2MPUCfQD'
                'fLV1apcyxe6msLE180NJ8UOZv1tSu+j/Fsz6lyP8ti31rwRyG43Z7ika+AS6'
                'BhqPXOSJAXoCby8G766CJ+SGiwfNM+iFfhE4r2Jw1hWc9Uu/9wjCevprir6Q'
                'jyetd0eB/Tge9xR9ITcbreqnz/ZP69eXG+VxiAjxNHlExaOG+tAnrHYQtOcZ'
                '1yg+lrnzwtQcR2RcXDqCA8h4jk7HYcbglOdwyjznPaajtzqntwr2D9ta9lS9'
                'j3xuX8fv+z2+xEl+X5Yvnwlf87S/MoezAk5o8ng3mOcbbwjtQqJtvzSHUwJV'
                'oGd/Q7Ev5Bo1HMoMP0z868R/2XFbmvbUeB8j6KTJNw4nrbipfjN3LK197YFi'
                'X8gtBh0burZh8y13tlzkmgdpxdOsmlae8cxqHoTxXHb9sMzk+019CAdIc7xq'
                'vJ5r6i2KVwK9l2nUGXsx+mnlSdR+UXcYcjzMJkAfJ++8TYLjse/GkJ7jsExP'
                '0z9Z7adZ72dJ+O/7jA75etyT+KfFIy9/JK0v/lV/JOE1cOHE60EinxeyyrOi'
                '5cdNz9dF9vPWS4tvkn2/SSBDKekvm7fOU5jpOUGtG94jQq6jF/o5aZybMXqP'
                'FP8KmXM1do0Dys6MjlU1DV6tG51n/Eefunw9zBT3ycbf79XuS+2yf/ZixhOV'
                'z1jUnwOX+iN9nKjzFO1+Qz02A5LHFX7jzkkyr0ogS7gGdmw0yhBPl39Fwa3M'
                '+fEKNor/om+cPd243dT4/1f7cTi686BoOEUdV1H5xa0zUfc71GcEO+iPhabI'
                'PIuGU9Rx3fK75bfMfpT2+qF+P5QW14tqXZ203jii0L68q5Tul+J4PVRwhFy3'
                'kcNwf9J08VC+W0haz4V4LdSj4kJ9mXu9qDrg9XhEXdamNchgF3pIPXf8BgAA'
                '//9jY4AAViDmAGJGIGaB0iDABOQJCfMI8wHZBkDMBlXHgkUfI5I+ASDLEEij'
                '6yPGPlYeZi6QPgsS9LFAxbfJ3pS7LOsuB9L/Akk/IwH9IPUaeNTDwGBVDwDn'
                '9/tU/r77zw==')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, E.ExportToDatabase))
        self.assertEqual(module.db_type, E.DB_MYSQL)
        self.assertEqual(module.db_name, "DefaultDB")
        self.assertEqual(module.sql_file_prefix, "SQL_")
        self.assertEqual(module.table_prefix, "Test")
        self.assertTrue(module.wants_agg_mean)
        self.assertFalse(module.wants_agg_median)
        self.assertTrue(module.wants_agg_std_dev)
        self.assertEqual(module.objects_choice, E.O_SELECT)
        self.assertEqual(len(module.objects_list.selections), 1)
        self.assertEqual(module.objects_list.selections[0], "Nuclei")
    
    def test_01_02_load_v9(self):
        data = ('eJztW91u2zYUpl0na9qhSHfTYkUBXg5DKshLXSTpxezE62Yg/lltdBi2YaMl'
                '2lEriwJFpXGHvccerY+wyz3CRFeKJVauaFlK7EUCBOXQ/M7Hc3gOeehY7cbg'
                'tHEMa4oK243Bk5FhYtgzERsROjmCFtuDJxQjhnVIrCM4cDHsagxW96F6ePS0'
                'drR/CL9R1UOQ7iq12ve8x+3nAGzzp3eX/Y+2fLkUurncx4wZ1tjZAhXw0G9/'
                '792vEDXQ0MSvkOliZ04RtLesERlM7cuP2kR3TdxBk3Bn7+q4kyGmTncUAP2P'
                'e8YFNvvGOyyYEHR7ic8NxyCWj/f1i62XvIQJvP0z8vYF9YYj6Of++fvR3D8l'
                'wT8V734cauf962DevxLjzzuh/ru+7LA3cELOjWBYMnp2BD1c7lJj3DtDTmAH'
                '13OQoGdb0MPljquZ2JDDbwl4Lren/R9Pff4fEvAPBTyXB/iCPXmJx66JKMQX'
                'NsUOn0jnuuw5wabprODPAXZY8/gDXk3AlyL4EtiX5I0bd1Xde6oCuXi6K+C5'
                '3KPERmPEPN/P2mXGfyui5xb42QvqtHnRJNAiDLpBQOfJvyifGpq35AF5/kpE'
                'T2U296uMOwlXjuDKoEPSz3eXOS783iRDZKa218v832Xy/oHAz+UmHiHXZLA1'
                'QWMMmwbFGiN0utK8L5tvVUm7s+JTwNXOlzhuda+60rjTxmde8yLGo6qo1evI'
                'v7zmQeSrqvnOQ1pc2nmIwdVk9r/PQDQ/uGxMxvpQ/div2wI+uAL8jv9Mu2+0'
                'LIYtx2DTJfTI7n9p6g/N9lTQFeqAbl/55Vf9z+pfv2Xul2XXgaCeSfLDbRDl'
                '57JOLGZj/CbL+jptvibxy8ZDkp7PBT1c9vfZ5rGiD2X1yMbFsvNZS+nHtOus'
                '7LourkMdYuE86884/w7eEqiZyHH8g+FVrtey9i7CJdV/9wV7ufwTNsZn/OuO'
                'c36wt7Rl8mxT5jkur18QiseUuJY+tzdJT9z61MemVzErirKC3xong8z8vcw+'
                'u+q5PI2/0pzvsrY7C1ye8+rVX2tp36K8rSfg0pyj1tneLPaDTbZvHfMxS7+I'
                '86fUrn+cWZ6f0tQ962xvnnXPOttdT7Bbtu7ZFHtv+rokc467knz7co4rCbi4'
                '/89dZXwb/HtsHuC2vJ649ZAMX3t1/VzRpqxroXFDw9KxnaO+Tcyr/yOuDrLJ'
                'k02xt8AVuAJX4ApcPvvFLojuF/ye7xcfyoBNsvem+TeJv6gHClweuDq43rgv'
                'cDcTVwdF3BW4Yt8rcAWuwBW4AlfgNgn3T2mOKwk4Lod/p8D7/xHiidvnvw71'
                '3/VlDZumTQl/348qk9lLaY5iEqTPCgBHOfX+nL0DEXofzE7gqQs89UU8ho4t'
                'ZoymNvXYXEYmiBma0vJbe15rI2jlvGcJvAcC70ESr4M1YumITi85+0GLDN8z'
                'ge/ZIj58YRPKGNERQ0PkYOW7WcOANP2Gj+NkJ4YvPN9lT3rw+P4Xn4ovAKJx'
                'NY+3f79Nw1eplHfugejv4+4m4CogGuezuAbLxfVXn+gf2LjO/Zf1c8m7/gOx'
                'RQ67')    
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, E.ExportToDatabase))
        self.assertEqual(module.db_type, E.DB_MYSQL)
        self.assertEqual(module.db_name, "TestDB")
        self.assertEqual(module.sql_file_prefix, "SQL_")
        self.assertEqual(module.table_prefix, "Test")
        self.assertEqual(module.db_host, "imgdb01")
        self.assertEqual(module.db_user, "cpuser")
        self.assertEqual(module.db_passwd, "dontpeek")
        self.assertFalse(module.store_csvs)
        self.assertTrue(module.wants_agg_mean)
        self.assertFalse(module.wants_agg_median)
        self.assertTrue(module.wants_agg_std_dev)
        self.assertEqual(module.objects_choice, E.O_SELECT)
        self.assertEqual(len(module.objects_list.selections), 1)
        self.assertEqual(module.objects_list.selections[0], "Cells")

    def make_workspace(self, wants_files):
        '''Make a measurements structure with image and object measurements'''
        class TestModule(cpm.CPModule):
            module_name = "TestModule"
            module_num = 1
            def __init__(self):
                self.image_name = cps.ImageNameProvider("Foo", IMAGE_NAME)
                self.objects_name = cps.ObjectNameProvider("Bar", OBJECT_NAME)
            
            def settings(self):
                return [self.image_name, self.objects_name]
            
            def get_measurement_columns(self, pipeline):
                return [(cpmeas.IMAGE, INT_IMG_MEASUREMENT, cpmeas.COLTYPE_INTEGER),
                        (cpmeas.IMAGE, FLOAT_IMG_MEASUREMENT, cpmeas.COLTYPE_FLOAT),
                        (cpmeas.IMAGE, STRING_IMG_MEASUREMENT, 
                         cpmeas.COLTYPE_VARCHAR_FORMAT % 40),
                        (cpmeas.IMAGE, OBJECT_COUNT_MEASUREMENT, cpmeas.COLTYPE_INTEGER),
                        (OBJECT_NAME, OBJ_MEASUREMENT, cpmeas.COLTYPE_FLOAT)]
            
            def get_categories(self, pipeline, object_name):
                return ([M_CATEGORY] 
                        if object_name == OBJECT_NAME
                        else [M_CATEGORY, "Count"] 
                        if object_name == cpmeas.IMAGE
                        else [])
            
            def get_measurements(self, pipeline, object_name, category):
                if category == M_CATEGORY:
                    if object_name == OBJECT_NAME:
                        return [OBJ_FEATURE]
                    else:
                        return [INT_IMG_FEATURE, FLOAT_IMG_FEATURE, STRING_IMG_FEATURE]
                elif category == "Count" and object_name == cpmeas.IMAGE:
                    return OBJECT_NAME
                return []
            
        m = cpmeas.Measurements()
        m.add_image_measurement(INT_IMG_MEASUREMENT, INT_VALUE)
        m.add_image_measurement(FLOAT_IMG_MEASUREMENT, FLOAT_VALUE)
        m.add_image_measurement(STRING_IMG_MEASUREMENT, STRING_VALUE)
        m.add_image_measurement(OBJECT_COUNT_MEASUREMENT, len(OBJ_VALUE))
        m.add_measurement(OBJECT_NAME, OBJ_MEASUREMENT, OBJ_VALUE)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        image_set.add(IMAGE_NAME, cpi.Image(np.zeros((10,10))))
        object_set = cpo.ObjectSet()
        objects = cpo.Objects()
        objects.segmented = np.array([[0,1,2,3],[0,1,2,3]])
        object_set.add_objects(objects, OBJECT_NAME)
        test_module = TestModule()
        pipeline = cpp.Pipeline()
        def callback_handler(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        
        pipeline.add_listener(callback_handler)
        pipeline.add_module(test_module)
        module = E.ExportToDatabase()
        module.module_num = 2
        table_prefix = "T_%s"%str(uuid.uuid4()).replace('-','')
        module.table_prefix.value = table_prefix
        module.want_table_prefix.value = True
        module.db_host.value = 'imgdb01'
        module.db_user.value = 'cpuser'
        module.db_passwd.value = 'cPus3r'
        module.db_name.value ='CPUnitTest'
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, image_set,
                                  object_set, m, image_set_list)
        for column in pipeline.get_measurement_columns():
            if (column[1].startswith("ModuleError_") or
                column[1].startswith("ExecutionTime_")):
                m.add_image_measurement(column[1],0)
        if wants_files:
            output_dir = tempfile.mkdtemp()
            module.use_default_output_directory.value = False
            module.output_directory.value = output_dir
            def finally_fn():
                for filename in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, filename))
            return workspace, module, output_dir, finally_fn
        else:
            return workspace, module
        
    def test_02_01_write_mysql_db(self):
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        os.chdir(output_dir)
        try:
            self.assertTrue(isinstance(module, E.ExportToDatabase))
            module.db_type = E.DB_MYSQL
            module.store_csvs.value = True
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.post_run(workspace)
            sql_file = os.path.join(output_dir, "SQL__SETUP.SQL")
            base_name = "SQL_1_1"
            image_file = os.path.join(output_dir, base_name+"_image.CSV")
            object_file = os.path.join(output_dir, base_name+"_object.CSV")
            for filename in (sql_file, image_file, object_file):
                self.assertTrue(os.path.isfile(filename))
            fd = open(sql_file,'rt')
            sql_text = fd.read()
            fd.close()
            for statement in sql_text.split(';'):
                if len(statement.strip()) == 0:
                    continue
                self.cursor.execute(statement)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = ("select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                         "from %s" %
                         (INT_IMG_MEASUREMENT, FLOAT_IMG_MEASUREMENT,
                          STRING_IMG_MEASUREMENT, OBJECT_NAME, image_table))
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            self.assertEqual(len(row), 5)
            self.assertEqual(row[0],1)
            self.assertAlmostEqual(row[1], INT_VALUE)
            self.assertAlmostEqual(row[2], FLOAT_VALUE)
            self.assertEqual(row[3], STRING_VALUE)
            self.assertEqual(row[4], len(OBJ_VALUE))
            self.assertRaises(StopIteration, self.cursor.next)
            statement = ("select ImageNumber, ObjectNumber, %s_%s "
                         "from %sPer_Object order by ObjectNumber"%
                         (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value))
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                self.assertEqual(len(row), 3)
                self.assertEqual(row[0], 1)
                self.assertEqual(row[1], i+1)
                self.assertAlmostEqual(row[2], value)
            self.assertRaises(StopIteration, self.cursor.next)
        finally:
            os.chdir(output_dir)
            finally_fn()
            for table_suffix in ("Per_Image","Per_Object"):
                table_name = module.table_prefix.value + table_suffix
                try:
                    self.cursor.execute("drop table %s.%s" %
                                        (module.db_name.value, table_name))
                except:
                    print "Failed to drop table %s"%table_name
    
    def test_02_02_mysql_direct(self):
        '''Write directly to the mysql DB, not to a file'''
        workspace, module = self.make_workspace(False)
        try:
            self.assertTrue(isinstance(module, E.ExportToDatabase))
            module.db_type = E.DB_MYSQL
            module.store_csvs.value = False
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.prepare_run(workspace.pipeline, workspace.image_set_list,None)
            module.prepare_group(workspace.pipeline, workspace.image_set_list,
                                 {}, [1])
            module.run(workspace)
            self.cursor.execute("use CPUnitTest")
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = ("select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                         "from %s" %
                         (INT_IMG_MEASUREMENT, FLOAT_IMG_MEASUREMENT,
                          STRING_IMG_MEASUREMENT, OBJECT_NAME, image_table))
            self.cursor.execute(statement)
            row = self.cursor.fetchone()
            self.assertEqual(len(row), 5)
            self.assertEqual(row[0],1)
            self.assertAlmostEqual(row[1], INT_VALUE)
            self.assertAlmostEqual(row[2], FLOAT_VALUE)
            self.assertEqual(row[3], STRING_VALUE)
            self.assertEqual(row[4], len(OBJ_VALUE))
            self.assertRaises(StopIteration, self.cursor.next)
            statement = ("select ImageNumber, ObjectNumber, %s_%s "
                         "from %sPer_Object order by ObjectNumber"%
                         (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value))
            self.cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = self.cursor.fetchone()
                self.assertEqual(len(row), 3)
                self.assertEqual(row[0], 1)
                self.assertEqual(row[1], i+1)
                self.assertAlmostEqual(row[2], value)
            self.assertRaises(StopIteration, self.cursor.next)
        finally:
            for table_suffix in ("Per_Image","Per_Object"):
                table_name = module.table_prefix.value + table_suffix
                try:
                    self.cursor.execute("drop table %s.%s" %
                                        (module.db_name.value, table_name))
                except:
                    print "Failed to drop table %s"%table_name
    
    def test_03_01_write_sqlite_direct(self):
        '''Write directly to a SQLite database'''
        workspace, module, output_dir, finally_fn = self.make_workspace(True)
        cursor = None
        connection = None
        try:
            self.assertTrue(isinstance(module, E.ExportToDatabase))
            module.db_type = E.DB_SQLITE
            module.store_csvs.value = False
            module.wants_agg_mean.value = False
            module.wants_agg_median.value = False
            module.wants_agg_std_dev.value = False
            module.objects_choice.value = E.O_ALL
            module.prepare_run(workspace.pipeline, workspace.image_set_list,None)
            module.prepare_group(workspace.pipeline, workspace.image_set_list,
                                 {}, [1])
            module.run(workspace)
            cursor, connection = self.get_sqlite_cursor(module)
            #
            # Now read the image file from the database
            #
            image_table = module.table_prefix.value + "Per_Image"
            statement = ("select ImageNumber, Image_%s, Image_%s, Image_%s, Image_Count_%s "
                         "from %s" %
                         (INT_IMG_MEASUREMENT, FLOAT_IMG_MEASUREMENT,
                          STRING_IMG_MEASUREMENT, OBJECT_NAME, image_table))
            cursor.execute(statement)
            row = cursor.fetchone()
            self.assertEqual(len(row), 5)
            self.assertEqual(row[0],1)
            self.assertAlmostEqual(row[1], INT_VALUE)
            self.assertAlmostEqual(row[2], FLOAT_VALUE)
            self.assertEqual(row[3], STRING_VALUE)
            self.assertEqual(row[4], len(OBJ_VALUE))
            self.assertRaises(StopIteration, cursor.next)
            statement = ("select ImageNumber, ObjectNumber, %s_%s "
                         "from %sPer_Object order by ObjectNumber"%
                         (OBJECT_NAME, OBJ_MEASUREMENT, module.table_prefix.value))
            cursor.execute(statement)
            for i, value in enumerate(OBJ_VALUE):
                row = cursor.fetchone()
                self.assertEqual(len(row), 3)
                self.assertEqual(row[0], 1)
                self.assertEqual(row[1], i+1)
                self.assertAlmostEqual(row[2], value)
            self.assertRaises(StopIteration, cursor.next)
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
            if hasattr(module, "cursor"):
                module.cursor.close()
            if hasattr(module, "connection"):
                module.connection.close()
            finally_fn()
        