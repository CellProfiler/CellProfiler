'''parameterhandler.py - Look for parameters in ImageJ plugins'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#
__version__="$Revision$"

import sys

import bioformats
import cellprofiler.utilities.jutil as J

from cellprofiler.preferences import get_headless

PARAMETER_HANDLER_CLASS = 'imagej/plugin/ParameterHandler'
RUNNABLE_CLASS = 'java/lang/Runnable'
ITERABLE_CLASS = 'java/lang/Iterable'
ITERATOR_CLASS = 'java/util/Iterator'
FIELD_CLASS = 'java/lang/reflect/Field'
PARAMETER_CLASS = 'imagej/plugin/Parameter'

'''Field type = integer'''
FT_INTEGER = "INTEGER"
'''Field type = floating point'''
FT_FLOAT = "FLOAT"
'''Field type = string'''
FT_STRING = "STRING"
'''Field type = image'''
FT_IMAGE = "IMAGE"
'''Field type = boolean'''
FT_BOOL = "BOOL"

def get_parameter_wrapper(parameter):
    '''wrap the imagej.plugin.Parameter class'''
    class Parameter(object):
        def __init__(self):
            self.o = parameter
        label = J.make_method('label', '()Ljava/lang/String;',
                              'Label to display in editor')
        digits = J.make_method('digits', '()I',
                               'Number of digits to display to right of decimal point')
        columns = J.make_method('columns', '()I',
                                'Number of columns in edit box')
        units = J.make_method('units', '()Ljava/lang/String;',
                              'Units to display to right of edit box')
        widget = J.make_method('widget', '()Ljava/lang/String;',
                             'Name of display widget')
        required = J.make_method('required', '()Z',
                                 'True if required, False if default available')
        persist = J.make_method('persist', '()Ljava/lang/String;',
                                'Key to use when persisting field value')
        output = J.make_method('output', '()Z',
                               'True if field is an output parameter, '
                               'False if field is an input parameter')
    return Parameter()
    
def get_input_fields_and_parameters(plugin):
    '''Get the input parameters from a plugin
    
    plugin - a Runnable plugin with @parameter annotations
    
    returns a dictionary of field name and wrapped parameter
    '''
    parameters = J.static_call(PARAMETER_HANDLER_CLASS,
                               'getInputParameters',
                               '(L%(RUNNABLE_CLASS)s;)L%(ITERABLE_CLASS)s;' %
                               globals(), plugin)
    return get_fields_and_parameters_from_iterator(parameters)

def get_output_fields_and_parameters(plugin):
    '''Get the output parameters from a plugin
    
    plugin - a Runnable plugin with @parameter annotations
    
    returns a dictionary of field name and wrapped parameter
    '''
    parameters = J.static_call(PARAMETER_HANDLER_CLASS,
                               'getOutputParameters',
                               '(L%(RUNNABLE_CLASS)s;)L%(ITERABLE_CLASS)s;' %
                               globals(), plugin)
    return get_fields_and_parameters_from_iterator(parameters)

def get_fields_and_parameters_from_iterator(parameters):
    iterator = J.call(parameters, 'iterator', 
                      '()L%(ITERATOR_CLASS)s;' % globals())
    
    result = []
    for jfield in J.iterate_java(iterator):
        field = J.get_field_wrapper(jfield)
        parameter = field.getAnnotation(PARAMETER_CLASS.replace('/','.'))
        if parameter is not None:
            parameter = get_parameter_wrapper(parameter)
            result.append((field, parameter))
    return result

field_mapping = {
    'ij.ImagePlus': FT_IMAGE,
    'java.lang.Byte': FT_INTEGER,
    'java.lang.Short': FT_INTEGER,
    'java.lang.Integer': FT_INTEGER,
    'java.lang.Long': FT_INTEGER,
    'int': FT_INTEGER,
    'short': FT_INTEGER,
    'byte': FT_INTEGER,
    'long': FT_INTEGER,
    'java.lang.Float': FT_FLOAT,
    'java.lang.Double': FT_FLOAT,
    'float': FT_FLOAT,
    'double': FT_FLOAT,
    'java.lang.String': FT_STRING,
    'java.lang.Boolean': FT_BOOL,
    'boolean': FT_BOOL }
    
def get_field_type(field):
    '''Determine the data type of the field
    
    field - a field returned from get_input/output_fields_and_parameters
    
    Returns one of FT_INTEGER, FT_FLOAT, FT_IMAGE, FT_BOOL
    '''
    t = field.getType()
    # t is a class itself, to wrap, we replace the class of class (= Class)
    # with t
    tc = J.get_class_wrapper(t)
    tc.o = t
    name = tc.getCanonicalName()
    return field_mapping.get(name, None)
    
