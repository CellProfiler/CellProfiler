# coding=utf-8

"""
Modules - pipeline processing modules for CellProfiler
"""

import logging

logger = logging.getLogger(__name__)
import re
import sys
import os.path
import glob
import cellprofiler.module as cpm
from cellprofiler.modules.plugins import plugin_list
from cellprofiler.preferences import get_plugin_directory

# python modules and their corresponding cellprofiler.module classes
pymodule_to_cpmodule = {'align': 'Align',
                        'calculatemath': 'CalculateMath',
                        'calculatestatistics': 'CalculateStatistics',
                        'classifyobjects': 'ClassifyObjects',
                        'closing': 'Closing',
                        'colortogray': 'ColorToGray',
                        'convertimagetoobjects': 'ConvertImageToObjects',
                        'convertobjectstoimage': 'ConvertObjectsToImage',
                        'correctilluminationcalculate': 'CorrectIlluminationCalculate',
                        'correctilluminationapply': 'CorrectIlluminationApply',
                        'createbatchfiles': 'CreateBatchFiles',
                        'crop': 'Crop',
                        'definegrid': 'DefineGrid',
                        'dilateimage': 'DilateImage',
                        'dilateobjects': 'DilateObjects',
                        'displaydensityplot': 'DisplayDensityPlot',
                        'displaydataonimage': 'DisplayDataOnImage',
                        'displayhistogram': 'DisplayHistogram',
                        'displayplatemap': 'DisplayPlatemap',
                        'displayscatterplot': 'DisplayScatterPlot',
                        'editobjectsmanually': 'EditObjectsManually',
                        'enhanceedges': 'EnhanceEdges',
                        'enhanceorsuppressfeatures': 'EnhanceOrSuppressFeatures',
                        'erosion': 'Erosion',
                        'expandorshrinkobjects': 'ExpandOrShrinkObjects',
                        'exporttodatabase': 'ExportToDatabase',
                        'exporttospreadsheet': 'ExportToSpreadsheet',
                        'fillobjects': 'FillObjects',
                        'filterobjects': 'FilterObjects',
                        'flagimage': 'FlagImage',
                        'flipandrotate': 'FlipAndRotate',
                        'gaussianfilter': 'GaussianFilter',
                        'graytocolor': 'GrayToColor',
                        'groups': 'Groups',
                        'identifydeadworms': 'IdentifyDeadWorms',
                        'identifyobjectsingrid': 'IdentifyObjectsInGrid',
                        'identifyobjectsmanually': 'IdentifyObjectsManually',
                        'identifyprimaryobjects': 'IdentifyPrimaryObjects',
                        'identifysecondaryobjects': 'IdentifySecondaryObjects',
                        'identifytertiaryobjects': 'IdentifyTertiaryObjects',
                        'imagemath': 'ImageMath',
                        'images': 'Images',
                        'invertforprinting': 'InvertForPrinting',
                        'labelimages': 'LabelImages',
                        'loadimages': 'LoadImages',
                        # 'loadimagesnew' : 'LoadImagesNew',
                        'loadsingleimage': 'LoadSingleImage',
                        'loaddata': 'LoadData',
                        'makeprojection': 'MakeProjection',
                        'maskimage': 'MaskImage',
                        'maskobjects': 'MaskObjects',
                        'measurecolocalization': 'MeasureColocalization',
                        'measuregranularity': 'MeasureGranularity',
                        'measureimageareaoccupied': 'MeasureImageAreaOccupied',
                        'measureimagegranularity': 'MeasureGranularity',
                        'measureimageintensity': 'MeasureImageIntensity',
                        'measureimageoverlap': 'MeasureImageOverlap',
                        'measureimagequality': 'MeasureImageQuality',
                        'measureimageskeleton': 'MeasureImageSkeleton',
                        'measureobjectintensity': 'MeasureObjectIntensity',
                        'measureobjectoverlap': 'MeasureObjectOverlap',
                        'measureobjectsizeshape': 'MeasureObjectSizeShape',
                        'measureobjectneighbors': 'MeasureObjectNeighbors',
                        'measureobjectintensitydistribution': 'MeasureObjectIntensityDistribution',
                        'measureobjectskeleton': 'MeasureObjectSkeleton',
                        'measuretexture': 'MeasureTexture',
                        'medialaxis': 'MedialAxis',
                        'medianfilter': 'MedianFilter',
                        'mergeoutputfiles': 'MergeOutputFiles',
                        'metadata': 'Metadata',
                        'morph': 'Morph',
                        'morphologicalskeleton': 'MorphologicalSkeleton',
                        'namesandtypes': 'NamesAndTypes',
                        'opening': 'Opening',
                        "overlayobjects": "OverlayObjects",
                        'overlayoutlines': 'OverlayOutlines',
                        'reducenoise': "ReduceNoise",
                        'relateobjects': 'RelateObjects',
                        'removeholes': 'RemoveHoles',
                        'rescaleintensity': 'RescaleIntensity',
                        'resize': 'Resize',
                        "resizeobjects": "ResizeObjects",
                        "savecroppedobjects": "SaveCroppedObjects",
                        'saveimages': 'SaveImages',
                        'shrinktoobjectcenters': 'ShrinkToObjectCenters',
                        'smooth': 'Smooth',
                        'splitormergeobjects': 'SplitOrMergeObjects',
                        'straightenworms': 'StraightenWorms',
                        'matchtemplate': 'MatchTemplate',
                        'threshold': 'Threshold',
                        'trackobjects': 'TrackObjects',
                        'tile': 'Tile',
                        'unmixcolors': 'UnmixColors',
                        'untangleworms': 'UntangleWorms',
                        'watershed': 'Watershed'
                        }

# the builtin CP modules that will be loaded from the cellprofiler.modules directory
builtin_modules = ['align',
                   'calculatemath',
                   'calculatestatistics',
                   'classifyobjects',
                   'closing',
                   'colortogray',
                   'convertimagetoobjects',
                   'convertobjectstoimage',
                   'correctilluminationcalculate',
                   'correctilluminationapply',
                   'createbatchfiles',
                   'crop',
                   'definegrid',
                   'dilateimage',
                   'dilateobjects',
                   'displaydataonimage',
                   'displaydensityplot',
                   'displayhistogram',
                   'displayplatemap',
                   'displayscatterplot',
                   'editobjectsmanually',
                   'enhanceedges',
                   'enhanceorsuppressfeatures',
                   'erosion',
                   'expandorshrinkobjects',
                   'exporttodatabase',
                   'exporttospreadsheet',
                   'fillobjects',
                   'filterobjects',
                   'flagimage',
                   'flipandrotate',
                   'gaussianfilter',
                   'graytocolor',
                   'groups',
                   'identifydeadworms',
                   'identifyobjectsingrid',
                   'identifyobjectsmanually',
                   'identifyprimaryobjects',
                   'identifysecondaryobjects',
                   'identifytertiaryobjects',
                   'imagemath',
                   'images',
                   'invertforprinting',
                   'labelimages',
                   'loadimages',
                   # 'loadimagesnew',
                   'loadsingleimage',
                   'loaddata',
                   'makeprojection',
                   'maskimage',
                   'maskobjects',
                   'medialaxis',
                   'metadata',
                   'measurecolocalization',
                   'measuregranularity',
                   'measureimageareaoccupied',
                   'measureimageintensity',
                   'measureimageoverlap',
                   'measureimagequality',
                   'measureimageskeleton',
                   'measureobjectintensity',
                   'measureobjectoverlap',
                   'measureobjectsizeshape',
                   'measureobjectneighbors',
                   'measureobjectintensitydistribution',
                   'measureobjectskeleton',
                   'measuretexture',
                   'medianfilter',
                   'mergeoutputfiles',
                   'morph',
                   'morphologicalskeleton',
                   'namesandtypes',
                   'opening',
                   "overlayobjects",
                   'overlayoutlines',
                   'reducenoise',
                   'relateobjects',
                   'removeholes',
                   'rescaleintensity',
                   'resizeobjects',
                   'resize',
                   'savecroppedobjects',
                   'saveimages',
                   'shrinktoobjectcenters',
                   'smooth',
                   'splitormergeobjects',
                   'straightenworms',
                   'matchtemplate',
                   "threshold",
                   'trackobjects',
                   'tile',
                   'unmixcolors',
                   'untangleworms',
                   'watershed'
                   ]

# Module renames and CP-Matlab to CP-python module substitutions
substitutions = {'Average': 'MakeProjection',
                 "ApplyThreshold": "Threshold",
                 'CalculateImageOverlap': 'MeasureImageOverlap',
                 'CalculateRatios': 'CalculateMath',
                 'ClassifyObjectsByTwoMeasurements': 'ClassifyObjects',
                 'Combine': 'ImageMath',
                 'cellprofiler.modules.converttoimage.ConvertToImage': 'ConvertObjectsToImage',
                 'ConvertToImage': 'ConvertObjectsToImage',
                 'CorrectIllumination_Apply': 'CorrectIlluminationApply',
                 'cellprofiler.modules.correctillumination_apply.CorrectIllumination_Apply': 'CorrectIlluminationApply',
                 'CorrectIllumination_Calculate': 'CorrectIlluminationCalculate',
                 'cellprofiler.modules.correctillumination_calculate.CorrectIllumination_Calculate': 'CorrectIlluminationCalculate',
                 'cellprofiler.modules.enhanceorsuppressspeckles.EnhanceOrSuppressSpeckles': 'EnhanceOrSuppressFeatures',
                 'CropObjects': 'SaveCroppedObjects',
                 'DifferentiateStains': 'UnmixColors',
                 'Dilation': 'DilateImage',
                 'EnhanceOrSuppressSpeckles': 'EnhanceOrSuppressFeatures',
                 'Exclude': 'MaskObjects',
                 'cellprofiler.modules.expandorshrink.ExpandOrShrink': 'ExpandOrShrinkObjects',
                 'ExpandOrShrink': 'ExpandOrShrinkObjects',
                 'ExportToExcel': 'ExportToSpreadsheet',
                 'cellprofiler.modules.exporttoexcel.ExportToExcel': 'ExportToSpreadsheet',
                 'FilterByObjectMeasurement': 'FilterObjects',
                 'cellprofiler.modules.filterbyobjectmeasurement.FilterByObjectMeasurement': 'FilterObjects',
                 'FindEdges': 'EnhanceEdges',
                 'cellprofiler.modules.findedges.FindEdges': 'EnhanceEdges',
                 'FlagImageForQC': 'FlagImage',
                 'Flip': 'FlipAndRotate',
                 'IdentifyPrimManual': 'IdentifyObjectsManually',
                 'cellprofiler.modules.identifytertiarysubregion.IdentifyTertiarySubregion': 'IdentifyTertiaryObjects',
                 'IdentifyTertiarySubregion': 'IdentifyTertiaryObjects',
                 'cellprofiler.modules.imageconvexhull.ImageConvexHull': 'Morph',
                 'ImageConvexHull': 'Morph',
                 'InvertIntensity': 'ImageMath',
                 'KeepLargestObject': 'FilterObjects',
                 'cellprofiler.modules.loadtext.LoadText': 'LoadData',
                 'LoadText': 'LoadData',
                 'cellprofiler.modules.measureimagegranularity.MeasureImageGranularity': 'MeasureGranularity',
                 'MeasureImageGranularity': 'MeasureGranularity',
                 'MeasureNeurons': 'MeasureObjectSkeleton',
                 'cellprofiler.modules.measureobjectareashape.MeasureObjectAreaShape': 'MeasureObjectSizeShape',
                 'MeasureObjectAreaShape': 'MeasureObjectSizeShape',
                 'MeasureCorrelation': 'MeasureColocalization',
                 'MeasureImageSaturationBlur': 'MeasureImageQuality',
                 'MeasureRadialDistribution': 'MeasureObjectIntensityDistribution',
                 'MeasureObjectRadialDistribution': 'MeasureObjectIntensityDistribution',
                 'cellprofiler.modules.measureobjectradialdistribution.MeasureObjectRadialDistribution': 'MeasureObjectIntensityDistribution',
                 'Multiply': 'ImageMath',
                 'NoiseReduction': "ReduceNoise",
                 'PlaceAdjacent': 'Tile',
                 'cellprofiler.modules.relate.Relate': 'RelateObjects',
                 'ReassignObjectNumbers': 'SplitOrMergeObjects',
                 'Relate': 'RelateObjects',
                 'Rotate': 'FlipAndRotate',
                 'SmoothOrEnhance': 'Smooth',
                 'SmoothKeepingEdges': 'Smooth',
                 'SplitIntoContiguousObjects': 'SplitOrMergeObjects',
                 'Subtract': 'ImageMath',
                 'UnifyObjects': 'SplitOrMergeObjects',
                 'cellprofiler.modules.overlay_outlines.OverlayOutlines': 'OverlayOutlines',
                 'CorrectIllumination_Apply': 'CorrectIlluminationApply',
                 'CorrectIllumination_Calculate': 'CorrectIlluminationCalculate'
                 }

all_modules = {}
svn_revisions = {}
pymodules = []
badmodules = []
datatools = []
pure_datatools = {}

do_not_override = ['set_settings', 'create_from_handles', 'test_valid', 'module_class']
should_override = ['create_settings', 'settings', 'run']


def check_module(module, name):
    if hasattr(module, 'do_not_check'):
        return
    assert name == module.module_name, "Module %s should have module_name %s (is %s)" % (name, name, module.module_name)
    for method_name in do_not_override:
        assert getattr(module, method_name) == getattr(cpm.Module,
                                                       method_name), "Module %s should not override method %s" % (
            name, method_name)
    for method_name in should_override:
        assert getattr(module, method_name) != getattr(cpm.Module,
                                                       method_name), "Module %s should override method %s" % (
            name, method_name)


def find_cpmodule(m):
    '''Returns the CPModule from within the loaded Python module

    m - an imported module

    returns the CPModule class
    '''
    for v, val in m.__dict__.iteritems():
        if isinstance(val, type) and issubclass(val, cpm.Module):
            return val
    raise ValueError("Could not find cellprofiler.module.Module class in %s" % m.__file__)


def fill_modules():
    del pymodules[:]
    del badmodules[:]
    del datatools[:]
    all_modules.clear()
    svn_revisions.clear()

    def add_module(mod, check_svn):
        try:
            m = __import__(mod, globals(), locals(), ['__all__'], 0)
            cp_module = find_cpmodule(m)
            name = cp_module.module_name
        except Exception as e:
            logger.warning("Could not load %s", mod, exc_info=True)
            badmodules.append((mod, e))
            return

        try:
            pymodules.append(m)
            if name in all_modules:
                logger.warning(
                        "Multiple definitions of module %s\n\told in %s\n\tnew in %s",
                        name, sys.modules[all_modules[name].__module__].__file__,
                        m.__file__)
            all_modules[name] = cp_module
            check_module(cp_module, name)
            # attempt to instantiate
            if not hasattr(cp_module, 'do_not_check'):
                cp_module()
            if hasattr(cp_module, "run_as_data_tool"):
                datatools.append(name)
            if check_svn and hasattr(m, '__version__'):
                match = re.match('^\$Revision: ([0-9]+) \$$', m.__version__)
                if match is not None:
                    svn_revisions[name] = match.groups()[0]
            if not hasattr(all_modules[name], "settings"):
                # No settings = pure data tool
                pure_datatools[name] = all_modules[name]
                del all_modules[name]
        except Exception as e:
            logger.warning("Failed to load %s", name, exc_info=True)
            badmodules.append((mod, e))
            if name in all_modules:
                del all_modules[name]
                del pymodules[-1]

    for mod in builtin_modules:
        add_module('cellprofiler.modules.' + mod, True)

    plugin_directory = get_plugin_directory()
    if plugin_directory is not None:
        old_path = sys.path
        sys.path.insert(0, plugin_directory)
        try:
            for mod in plugin_list():
                add_module(mod, False)
        finally:
            sys.path = old_path

    datatools.sort()
    if len(badmodules) > 0:
        logger.warning("could not load these modules: %s",
                       ",".join([x[0] for x in badmodules]))


def add_module_for_tst(module_class):
    all_modules[module_class.module_name] = module_class


fill_modules()

__all__ = ['instantiate_module', 'get_module_names', 'reload_modules',
           'add_module_for_tst', 'builtin_modules']

replaced_modules = {
    'LoadImageDirectory': ['LoadImages', 'LoadData'],
    'GroupMovieFrames': ['LoadImages'],
    'IdentifyPrimLoG': ['IdentifyPrimaryObjects'],
    'FileNameMetadata': ['LoadImages']
}
depricated_modules = [
    'CorrectIllumination_Calculate_kate',
    'SubtractBackground'
]
unimplemented_modules = [
    'LabelImages', 'Restart', 'SplitOrSpliceMovie'
]


def get_module_class(module_name):
    if module_name in substitutions:
        module_name = substitutions[module_name]
    module_class = module_name.split('.')[-1]
    if module_class not in all_modules:
        if module_class in pure_datatools:
            return pure_datatools[module_class]
        if module_class in unimplemented_modules:
            raise ValueError(("The %s module has not yet been implemented. "
                              "It will be available in a later version "
                              "of CellProfiler.") % module_class)
        if module_class in depricated_modules:
            raise ValueError(("The %s module has been deprecated and will "
                              "not be implemented in CellProfiler 2.0.") %
                             module_class)
        if module_class in replaced_modules:
            raise ValueError(("The %s module no longer exists. You can find "
                              "similar functionality in: %s") %
                             (module_class, ", ".join(replaced_modules[module_class])))
        raise ValueError("Could not find the %s module" % module_class)
    return all_modules[module_class]


def instantiate_module(module_name):
    module = get_module_class(module_name)()
    if module_name in svn_revisions:
        module.svn_version = svn_revisions[module_name]
    return module


def get_module_names():
    return all_modules.keys()


def get_data_tool_names():
    return datatools


def reload_modules():
    for m in pymodules:
        try:
            del sys.modules[m.__name__]
        except:
            pass
    fill_modules()
