"""Modules - pipeline processing modules for CellProfiler
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
                        'activecontourmodel': 'ActiveContourModel',
                        'applythreshold': 'ApplyThreshold',
                        'blobdetection': 'BlobDetection',
                        'calculateimageoverlap': 'CalculateImageOverlap',
                        'calculatemath': 'CalculateMath',
                        'calculatestatistics': 'CalculateStatistics',
                        'classifyobjects': 'ClassifyObjects',
                        'closing': 'Closing',
                        'colortogray': 'ColorToGray',
                        'convertobjectstoimage': 'ConvertObjectsToImage',
                        'correctilluminationcalculate': 'CorrectIlluminationCalculate',
                        'correctilluminationapply': 'CorrectIlluminationApply',
                        'createbatchfiles': 'CreateBatchFiles',
                        'crop': 'Crop',
                        "cropobjects": "CropObjects",
                        'definegrid': 'DefineGrid',
                        'dilation': 'Dilation',
                        'displaydensityplot': 'DisplayDensityPlot',
                        'displaydataonimage': 'DisplayDataOnImage',
                        'displayhistogram': 'DisplayHistogram',
                        'displayplatemap': 'DisplayPlatemap',
                        'displayscatterplot': 'DisplayScatterPlot',
                        'editobjectsmanually': 'EditObjectsManually',
                        'edgedetection': 'EdgeDetection',
                        'enhanceedges': 'EnhanceEdges',
                        'enhanceorsuppressfeatures': 'EnhanceOrSuppressFeatures',
                        'erosion': 'Erosion',
                        'expandorshrinkobjects': 'ExpandOrShrinkObjects',
                        'exporttodatabase': 'ExportToDatabase',
                        'exporttospreadsheet': 'ExportToSpreadsheet',
                        'filterobjects': 'FilterObjects',
                        'flagimage': 'FlagImage',
                        'flipandrotate': 'FlipAndRotate',
                        'gammacorrection': 'GammaCorrection',
                        'gaussianfilter': 'GaussianFilter',
                        'graytocolor': 'GrayToColor',
                        'groups': 'Groups',
                        'histogramequalization': 'HistogramEqualization',
                        'identifydeadworms': 'IdentifyDeadWorms',
                        'identifyobjectsingrid': 'IdentifyObjectsInGrid',
                        'identifyobjectsmanually': 'IdentifyObjectsManually',
                        'identifyprimaryobjects': 'IdentifyPrimaryObjects',
                        'identifysecondaryobjects': 'IdentifySecondaryObjects',
                        'identifytertiaryobjects': 'IdentifyTertiaryObjects',
                        'imagegradient': 'ImageGradient',
                        'imagemath': 'ImageMath',
                        'images': 'Images',
                        'invertforprinting': 'InvertForPrinting',
                        'labelimages': 'LabelImages',
                        'laplacianofgaussian': 'LaplacianOfGaussian',
                        'loadimages': 'LoadImages',
                        # 'loadimagesnew' : 'LoadImagesNew',
                        'loadsingleimage': 'LoadSingleImage',
                        'loaddata': 'LoadData',
                        'makeprojection': 'MakeProjection',
                        'maskimage': 'MaskImage',
                        'maskobjects': 'MaskObjects',
                        'measurecorrelation': 'MeasureCorrelation',
                        'measuregranularity': 'MeasureGranularity',
                        'measureimageareaoccupied': 'MeasureImageAreaOccupied',
                        'measureimagegranularity': 'MeasureGranularity',
                        'measureimageintensity': 'MeasureImageIntensity',
                        'measureimagequality': 'MeasureImageQuality',
                        'measureobjectintensity': 'MeasureObjectIntensity',
                        'measureobjectsizeshape': 'MeasureObjectSizeShape',
                        'measureobjectneighbors': 'MeasureObjectNeighbors',
                        'measureobjectintensitydistribution': 'MeasureObjectIntensityDistribution',
                        'measureneurons': 'MeasureNeurons',
                        'measuretexture': 'MeasureTexture',
                        'medianfilter': 'MedianFilter',
                        'medialaxis': 'MedialAxis',
                        'mergeoutputfiles': 'MergeOutputFiles',
                        'metadata': 'Metadata',
                        'morph': 'Morph',
                        'morphologicalskeleton': 'MorphologicalSkeleton',
                        'namesandtypes': 'NamesAndTypes',
                        'opening': 'Opening',
                        'noisereduction': 'NoiseReduction',
                        "overlayobjects": "OverlayObjects",
                        'overlayoutlines': 'OverlayOutlines',
                        'randomwalkeralgorithm': 'RandomWalkerAlgorithm',
                        'relateobjects': 'RelateObjects',
                        'reassignobjectnumbers': 'ReassignObjectNumbers',
                        'removeholes': 'RemoveHoles',
                        'removeobjects': 'RemoveObjects',
                        'rescaleintensity': 'RescaleIntensity',
                        'resize': 'Resize',
                        "resizeobjects": "ResizeObjects",
                        'save': 'Save',
                        'saveimages': 'SaveImages',
                        'smooth': 'Smooth',
                        'straightenworms': 'StraightenWorms',
                        'matchtemplate': 'MatchTemplate',
                        'trackobjects': 'TrackObjects',
                        'tile': 'Tile',
                        'tophattransform': 'TopHatTransform',
                        'calculateimageoverlap': 'CalculateImageOverlap',
                        'unmixcolors': 'UnmixColors',
                        'untangleworms': 'UntangleWorms',
                        'watershed': 'Watershed'
                        }

# the builtin CP modules that will be loaded from the cellprofiler.modules directory
builtin_modules = ['align',
                   'activecontourmodel',
                   'applythreshold',
                   'blobdetection',
                   'calculateimageoverlap',
                   'calculatemath',
                   'calculatestatistics',
                   'classifyobjects',
                   'closing',
                   'colortogray',
                   'convertobjectstoimage',
                   'correctilluminationcalculate',
                   'correctilluminationapply',
                   'createbatchfiles',
                   'crop',
                   'cropobjects',
                   'definegrid',
                   'dilation',
                   'displaydataonimage',
                   'displaydensityplot',
                   'displayhistogram',
                   'displayplatemap',
                   'displayscatterplot',
                   'editobjectsmanually',
                   'edgedetection',
                   'enhanceedges',
                   'enhanceorsuppressfeatures',
                   'erosion',
                   'expandorshrinkobjects',
                   'exporttodatabase',
                   'exporttospreadsheet',
                   'filterobjects',
                   'flagimage',
                   'flipandrotate',
                   'gammacorrection',
                   'gaussianfilter',
                   'graytocolor',
                   'groups',
                   'histogramequalization',
                   'identifydeadworms',
                   'identifyobjectsingrid',
                   'identifyobjectsmanually',
                   'identifyprimaryobjects',
                   'identifysecondaryobjects',
                   'identifytertiaryobjects',
                   'imagegradient',
                   'imagemath',
                   'images',
                   'invertforprinting',
                   'labelimages',
                   'laplacianofgaussian',
                   'loadimages',
                   # 'loadimagesnew',
                   'loadsingleimage',
                   'loaddata',
                   'makeprojection',
                   'maskimage',
                   'maskobjects',
                   'metadata',
                   'measurecorrelation',
                   'measuregranularity',
                   'measureimageareaoccupied',
                   'measureimageintensity',
                   'measureimagequality',
                   'measureobjectintensity',
                   'measureobjectsizeshape',
                   'measureobjectneighbors',
                   'measureobjectintensitydistribution',
                   'measureneurons',
                   'measuretexture',
                   'medianfilter',
                   'medialaxis',
                   'mergeoutputfiles',
                   'morph',
                   'morphologicalskeleton',
                   'namesandtypes',
                   'opening',
                   'noisereduction',
                   "overlayobjects",
                   'overlayoutlines',
                   'randomwalkeralgorithm',
                   'relateobjects',
                   'reassignobjectnumbers',
                   'removeholes',
                   'removeobjects',
                   'rescaleintensity',
                   'resize',
                   "resizeobjects",
                   'save',
                   'saveimages',
                   'smooth',
                   'straightenworms',
                   'matchtemplate',
                   'trackobjects',
                   'tile',
                   'tophattransform',
                   'unmixcolors',
                   'untangleworms',
                   'watershed'
                   ]

# CP-Matlab to CP-python module substitutions
substitutions = {'Average': 'MakeProjection',
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
                 'DifferentiateStains': 'UnmixColors',
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
                 'cellprofiler.modules.measureobjectareashape.MeasureObjectAreaShape': 'MeasureObjectSizeShape',
                 'MeasureObjectAreaShape': 'MeasureObjectSizeShape',
                 'MeasureImageSaturationBlur': 'MeasureImageQuality',
                 'MeasureRadialDistribution': 'MeasureObjectIntensityDistribution',
                 'MeasureObjectRadialDistribution': 'MeasureObjectIntensityDistribution',
                 'cellprofiler.modules.measureobjectradialdistribution.MeasureObjectRadialDistribution': 'MeasureObjectIntensityDistribution',
                 'Multiply': 'ImageMath',
                 'PlaceAdjacent': 'Tile',
                 'cellprofiler.modules.relabelobjects.RelabelObjects': 'ReassignObjectNumbers',
                 'RelabelObjects': 'ReassignObjectNumbers',
                 'cellprofiler.modules.relate.Relate': 'RelateObjects',
                 'Relate': 'RelateObjects',
                 'Rotate': 'FlipAndRotate',
                 'SmoothOrEnhance': 'Smooth',
                 'SmoothKeepingEdges': 'Smooth',
                 'SplitIntoContiguousObjects': 'ReassignObjectNumbers',
                 'Subtract': 'ImageMath',
                 'UnifyObjects': 'ReassignObjectNumbers',
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
        except Exception, e:
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
        except Exception, e:
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
           'output_module_html', 'add_module_for_tst', 'builtin_modules']

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
    if not all_modules.has_key(module_class):
        if pure_datatools.has_key(module_class):
            return pure_datatools[module_class]
        if module_class in unimplemented_modules:
            raise ValueError(("The %s module has not yet been implemented. "
                              "It will be available in a later version "
                              "of CellProfiler.") % module_class)
        if module_class in depricated_modules:
            raise ValueError(("The %s module has been deprecated and will "
                              "not be implemented in CellProfiler 2.0.") %
                             module_class)
        if replaced_modules.has_key(module_class):
            raise ValueError(("The %s module no longer exists. You can find "
                              "similar functionality in: %s") %
                             (module_class, ", ".join(replaced_modules[module_class])))
        raise ValueError("Could not find the %s module" % module_class)
    return all_modules[module_class]


def instantiate_module(module_name):
    module = get_module_class(module_name)()
    if svn_revisions.has_key(module_name):
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
