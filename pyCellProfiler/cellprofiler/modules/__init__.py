"""Modules - pipeline processing modules for CellProfiler

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import re
from cellprofiler.cpmodule import CPModule

# python modules and their corresponding cellprofiler.module classes
pymodule_to_cpmodule = {'align' : 'Align',
                        'applythreshold' : 'ApplyThreshold',
                        'calculatemath' : 'CalculateMath',
                        'calculatestatistics' : 'CalculateStatistics',
                        'classifyobjects' : 'ClassifyObjects',
                        'colortogray' : 'ColorToGray',
                        'conservememory' : 'ConserveMemory',
                        'convertobjectstoimage' : 'ConvertObjectsToImage',
                        'correctillumination_calculate' : 'CorrectIllumination_Calculate',
                        'correctillumination_apply' : 'CorrectIllumination_Apply',
                        'createbatchfiles' : 'CreateBatchFiles',
                        'crop' : 'Crop',
                        'definegrid' : 'DefineGrid',
                        'density' : 'DensityPlot',
                        'displaydataonimage' : 'DisplayDataOnImage',
                        'enhanceedges' : 'EnhanceEdges',
                        'enhanceorsuppressfeatures' : 'EnhanceOrSuppressFeatures',
                        'expandorshrinkobjects' : 'ExpandOrShrinkObjects',
                        'exporttodatabase' : 'ExportToDatabase',
                        'exporttospreadsheet' : 'ExportToSpreadsheet',
                        'filterobjects' : 'FilterObjects',
                        'flagimage' : 'FlagImage',
                        'flipandrotate' : 'FlipAndRotate',
                        'graytocolor' : 'GrayToColor',
                        'histogram' : 'Histogram',
                        'identifyobjectsingrid': 'IdentifyObjectsInGrid',
                        'identifyprimaryobjects' : 'IdentifyPrimaryObjects',
                        'identifysecondaryobjects' : 'IdentifySecondaryObjects',
                        'identifytertiaryobjects' : 'IdentifyTertiaryObjects',
                        'imagemath' : 'ImageMath',
                        'invertforprinting' : 'InvertForPrinting',
                        'loadimages' : 'LoadImages',
                        'loadimagesnew' : 'LoadImagesNew',
                        'loadsingleimage' : 'LoadSingleImage',
                        'loaddata' : 'LoadData',
                        'makeprojection' : 'MakeProjection',
                        'maskimage' : 'MaskImage',
                        'measurecorrelation' : 'MeasureCorrelation',
                        'measureimageareaoccupied' : 'MeasureImageAreaOccupied',
                        'measureimagegranularity' : 'MeasureImageGranularity',
                        'measureimageintensity' : 'MeasureImageIntensity',
                        'measureimagequality' : 'MeasureImageQuality',
                        'measureobjectintensity' : 'MeasureObjectIntensity',
                        'measureobjectsizeshape' : 'MeasureObjectSizeShape',
                        'measureobjectneighbors' : 'MeasureObjectNeighbors',
                        'measureobjectradialdistribution' : 'MeasureObjectRadialDistribution',
                        'measureneurons': 'MeasureNeurons',
                        'measuretexture' : 'MeasureTexture',
                        'morph' : 'Morph',
                        'overlay_outlines' : 'OverlayOutlines',
                        'pausecellprofiler': 'PauseCellProfiler',
                        'relateobjects' : 'RelateObjects',
                        'reassignobjectnumbers': 'ReassignObjectNumbers',
                        'rescaleintensity' : 'RescaleIntensity',
                        'resize' : 'Resize',
                        'saveimages' : 'SaveImages',
                        'scatter' : 'ScatterPlot',
                        'smooth' : 'Smooth',
                        'trackobjects' : 'TrackObjects'
                        }

# CP-Matlab to CP-python module substitutions
substitutions = {'Average': 'MakeProjection',
                 'CalculateRatios': 'CalculateMath',
                 'ClassifyObjectsByTwoMeasurements' : 'ClassifyObjects',
                 'Combine': 'ImageMath',
                 'cellprofiler.modules.converttoimage.ConvertToImage': 'ConvertObjectsToImage',
                 'ConvertToImage': 'ConvertObjectsToImage',
                 'cellprofiler.modules.enhanceorsuppressspeckles.EnhanceOrSuppressSpeckles': 'EnhanceOrSuppressFeatures',
                 'EnhanceOrSuppressSpeckles': 'EnhanceOrSuppressFeatures',
                 'cellprofiler.modules.expandorshrink.ExpandOrShrink': 'ExpandOrShrinkObjects',
                 'ExpandOrShrink':'ExpandOrShrinkObjects',
                 'ExportToExcel': 'ExportToSpreadsheet',
                 'cellprofiler.modules.exporttoexcel.ExportToExcel': 'ExportToSpreadsheet',
                 'FilterByObjectMeasurement': 'FilterObjects',
                 'cellprofiler.modules.filterbyobjectmeasurement.FilterByObjectMeasurement': 'FilterObjects',
                 'FindEdges':'EnhanceEdges',
                 'cellprofiler.modules.findedges.FindEdges':'EnhanceEdges',
                 'FlagImageForQC' : 'FlagImage',
                 'Flip' : 'FlipAndRotate',
                 'cellprofiler.modules.identifyprimautomatic.IdentifyPrimAutomatic': 'IdentifyPrimaryObjects',
                 'IdentifyPrimAutomatic':'IdentifyPrimaryObjects',
                 'cellprofiler.modules.identifysecondary.IdentifySecondary': 'IdentifySecondaryObjects',
                 'IdentifySecondary': 'IdentifySecondaryObjects',
                 'cellprofiler.modules.identifytertiarysubregion.IdentifyTertiarySubregion': 'IdentifyTertiaryObjects',
                 'IdentifyTertiarySubregion': 'IdentifyTertiaryObjects',
                 'cellprofiler.modules.imageconvexhull.ImageConvexHull': 'Morph',
                 'ImageConvexHull': 'Morph',
                 'InvertIntensity': 'ImageMath',
                 'KeepLargestObject' : 'FilterObjects',
                 'cellprofiler.modules.loadtext.LoadText': 'LoadData',
                 'LoadText': 'LoadData',
                 'cellprofiler.modules.measureobjectareashape.MeasureObjectAreaShape':'MeasureObjectSizeShape',
                 'MeasureObjectAreaShape':'MeasureObjectSizeShape',
                 'MeasureImageSaturationBlur': 'MeasureImageQuality',
                 'MeasureRadialDistribution' : 'MeasureObjectRadialDistribution',
                 'Multiply': 'ImageMath',
                 'cellprofiler.modules.relabelobjects.RelabelObjects':'ReassignObjectNumbers',
                 'RelabelObjects': 'ReassignObjectNumbers',
                 'cellprofiler.modules.relate.Relate': 'RelateObjects',
                 'Relate': 'RelateObjects',
                 'Rotate' : 'FlipAndRotate',
                 'SmoothOrEnhance' : 'Smooth',
                 'SmoothKeepingEdges' : 'Smooth',
                 'cellprofiler.modules.speedupcellprofiler.SpeedUpCellProfiler':'ConserveMemory',
                 'SpeedUpCellProfiler':'ConserveMemory',
                 'SplitIntoContiguousObjects': 'ReassignObjectNumbers',
                 'Subtract': 'ImageMath',
                 'UnifyObjects': 'ReassignObjectNumbers'
                 }

all_modules = {}
svn_revisions = {}
pymodules = []
badmodules = []
datatools = []

do_not_override = ['__init__', 'set_settings', 'create_from_handles', 'test_valid', 'module_class']
should_override = ['create_settings', 'settings', 'run']

def check_module(module, name):
    if hasattr(module, 'do_not_check'):
        return
    assert name == module.module_name, "Module %s should have module_name %s (is %s)"%(name, name, module.module_name)
    for method_name in do_not_override:
        assert getattr(module, method_name) == getattr(CPModule, method_name), "Module %s should not override method %s"%(name, method_name)
    for method_name in should_override:
        assert getattr(module, method_name) != getattr(CPModule, method_name), "Module %s should override method %s"%(name, method_name)
    

def fill_modules():
    del pymodules[:]
    del badmodules[:]
    del datatools[:]
    all_modules.clear()
    svn_revisions.clear()
    for mod, name in pymodule_to_cpmodule.items():
        try:
            m = __import__('cellprofiler.modules.' + mod, globals(), locals(), [name])
            assert not name in all_modules, "Module %s appears more than once in module list"%(name)
        except Exception, e:
            import traceback
            print traceback.print_exc(e)
            badmodules.append((mod, e))
            continue
        
        try:
            pymodules.append(m)
            all_modules[name] = m.__dict__[name]
            check_module(m.__dict__[name], name)
            # attempt to instantiate
            all_modules[name]()
            if hasattr(all_modules[name], "run_as_data_tool"):
                datatools.append(name)
            if hasattr(m, '__version__'):
                match = re.match('^\$Revision: ([0-9]+) \$$', m.__version__)
                if match is not None:
                    svn_revisions[name] = match.groups()[0]
        except Exception, e:
            import traceback
            print traceback.print_exc(e)
            badmodules.append((mod, e))
            if name in all_modules:
                del all_modules[name]
                del pymodules[-1]
    datatools.sort()
    if len(badmodules) > 0:
        print "could not load these modules", badmodules
        
fill_modules()
    
__all__ = ['instantiate_module', 'get_module_classes', 'reload_modules']

def instantiate_module(module_name):
    if module_name in substitutions: 
        module_name = substitutions[module_name]
    module_class = module_name.split('.')[-1]
    if not all_modules.has_key(module_class):
        raise ValueError("Could not find the %s module"%module_class)
    module = all_modules[module_class]()
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
            reload(m)
        except:
            pass
    fill_modules()
