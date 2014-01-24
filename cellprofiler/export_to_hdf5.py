#!/usr/bin/env python
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

"""
Export per-object features from database to an HDF5 file.

"""

import sys
import getpass
import decimal
import types
import MySQLdb
from MySQLdb.cursors import SSCursor
import progressbar
from cellprofiler.utilities.hdf5_dict import HDF5Dict, get_top_level_group
from cellprofiler.utilities.hdf5_dict import VERSION
from cellprofiler.measurements import Measurements
import cellprofiler.preferences

cellprofiler.preferences.set_headless()

def result_names(cursor):
    return [name for (name, type_code, display_size, internal_size, precision, 
                      scale, null_ok) in cursor.description]

def result_dtype(cursor):
    """
    Return an appropriate descriptor for a numpy array in which the
    result can be stored.

    """
    descr = []
    for (name, type_code, display_size, internal_size, precision, 
         scale, null_ok), flags in zip(cursor.description, 
                                       cursor.description_flags):
        conversion = cursor.connection.converter[type_code]
        if isinstance(conversion, list):
            fun2 = None
            for mask, fun in conversion:
                fun2 = fun
                if mask & flags:
                    break
        else:
            fun2 = conversion
        if fun2 in [decimal.Decimal, types.FloatType]:
            dtype = 'f8'
        elif fun2 in [types.IntType, types.LongType]:
            dtype = 'i4'
        elif fun2 in [types.StringType]:
            dtype = '|S%d'%(internal_size,)
        descr.append((name, dtype))
    return descr


database_and_table, filename = sys.argv[1:]
database_name, table_name = database_and_table.split('.')

connection = MySQLdb.connect(host='imgdb02', user='cpuser', 
                             passwd=getpass.getpass(), db=database_name)
cursor = SSCursor(connection)
cursor.execute('select count(*) from %s' % table_name)
nrows, = cursor.fetchall()[0]
print nrows, 'rows'
cursor.execute('SELECT * from %s' % table_name)
colnames = result_names(cursor)

m = Measurements(filename)


object_names = set(name.split('_', 1)[0]
                   for name in colnames
                   if '_' in name)
print 'Object names:', ', '.join(object_names)

columns = []
for i, (name, dtype) in enumerate(result_dtype(cursor)):
    if name == 'ImageNumber':
        image_number_index = i
    elif name == 'ObjectNumber':
        for on in object_names:
            columns.append((on, 'Number_Object_Number', dtype))
    else:
        object_name, feature_name = name.split('_', 1)
        columns.append((object_name, feature_name, dtype))
m.initialize(columns)


progress = progressbar.ProgressBar(widgets=[progressbar.Percentage(), ' ', 
                                            progressbar.Bar(), ' ', 
                                            progressbar.ETA()],
                                   maxval=nrows)

for row_number, row in progress(enumerate(cursor)):
    assert isinstance(row[image_number_index], long)
    image_set_number = int(row[image_number_index])
    for name, data in zip(colnames, row):
        if name == 'ImageNumber':
            pass
        elif name == 'ObjectNumber':
            for on in object_names:
                m.add_measurement(on, 'Number_Object_Number', [data],
                                  image_set_number=image_set_number)
        else:
            object_name, feature_name = name.split('_', 1)
            m.add_measurement(object_name, feature_name, [data], 
                              image_set_number=image_set_number)



