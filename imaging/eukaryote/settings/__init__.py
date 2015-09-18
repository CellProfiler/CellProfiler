"""
setting.py - represents a module setting
"""

import logging

logger = logging.getLogger(__name__)
import json
import matplotlib.cm
import numpy as np
import os
import sys
import re
import uuid

from cellprofiler.preferences import \
    DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, \
    DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, \
    ABSOLUTE_FOLDER_NAME, URL_FOLDER_NAME, NO_FOLDER_NAME, \
    get_default_image_directory, get_default_output_directory, \
    standardize_default_folder_names
import cellprofiler.measurements

from cellprofiler.utilities.utf16encode import utf16encode

'''Matlab CellProfiler uses this string for settings to be excluded'''
DO_NOT_USE = 'Do not use'
'''Matlab CellProfiler uses this string for automatically calculated settings'''
AUTOMATIC = "Automatic"
'''Value to store for boolean True settings'''
YES = 'Yes'
'''Value to store for boolean False settings'''
NO = 'No'
LEAVE_BLANK = 'Leave blank'
DEFAULT = 'Default'
NONE = 'None'

'''Names providers and subscribers of images'''
IMAGE_GROUP = 'imagegroup'

'''Names providers and subscribers of objects'''
OBJECT_GROUP = 'objectgroup'

MEASUREMENTS_GROUP = 'measurementsgroup'

'''Names providers and subscribers of grid information'''
GRID_GROUP = 'gridgroup'

'''Indicates that the image comes from a cropping operation'''
CROPPING_ATTRIBUTE = "cropping_image"
'''Indicates that the image was loaded from a file and has a file name and path'''
FILE_IMAGE_ATTRIBUTE = "file_image"
'''Indicates that the image is external (eg: from Java)'''
EXTERNAL_IMAGE_ATTRIBUTE = "external_image"
'''Indicates that the image is the result of an aggregate operation'''
AGGREGATE_IMAGE_ATTRIBUTE = "aggregate_image"
'''Indicates that the image is only available on the last cycle'''
AVAILABLE_ON_LAST_ATTRIBUTE = "available_on_last"
'''Indicates that the control can contain metadata tags'''
METADATA_ATTRIBUTE = "metadata"

SUPPORT_URLS_SHOW_DIR = "show_directory"

BEGIN = "begin"
END = "end"

PROVIDED_ATTRIBUTES = "provided_attributes"

REQUIRED_ATTRIBUTES = "required_attributes"


def filter_duplicate_names(name_list):
    '''remove any repeated names from a list of (name, ...) keeping the last occurrence.'''
    name_dict = dict(zip((n[0] for n in name_list), name_list))
    return [name_dict[n[0]] for n in name_list]


def get_name_provider_choices(pipeline, last_setting, group):
    '''Scan the pipeline to find name providers for the given group
    
    pipeline - pipeline to scan
    last_setting - scan the modules in order until you arrive at this setting
    group - the name of the group of providers to scan
    returns a list of tuples, each with (provider name, module name, module number)
    '''
    choices = []
    for module in pipeline.modules(False):
        module_choices = [
            (other_name, module.module_name, module.module_num,
             module.is_input_module())
            for other_name in module.other_providers(group)]
        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                return filter_duplicate_names(choices)
            if (isinstance(setting, NameProvider) and
                    module.enabled and
                        setting != DO_NOT_USE and
                    last_setting.matches(setting)):
                module_choices.append((
                    setting.value, module.module_name, module.module_num,
                    module.is_input_module()))
        choices += module_choices
    assert False, "Setting not among visible settings in pipeline"


def get_name_providers(pipeline, last_setting):
    '''Scan the pipeline to find name providers matching the name given in the setting
    
    pipeline - pipeline to scan
    last_setting - scan the modules in order until you arrive at this setting
    returns a list of providers that provide a correct "thing" with the
    same name as that of the subscriber
    '''
    choices = []
    for module in pipeline.modules(False):
        module_choices = []
        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                return choices
            if (isinstance(setting, NameProvider) and
                        setting != DO_NOT_USE and
                    module.enabled and
                    last_setting.matches(setting) and
                        setting.value == last_setting.value):
                module_choices.append(setting)
        choices += module_choices
    assert False, "Setting not among visible settings in pipeline"
