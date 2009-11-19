"""Modules - pipeline processing modules for CellProfiler

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

from cellprofiler.cpmodule import CPModule

# python modules and their corresponding cellprofiler.module classes
pymodule_to_cpmodule = {'align' : 'Align',
                        'applythreshold' : 'ApplyThreshold',
                        'calculatemath' : 'CalculateMath',
                        'calculatestatistics' : 'CalculateStatistics',
                        'colortogray' : 'ColorToGray',
                        'converttoimage' : 'ConvertToImage',
                        'correctillumination_calculate' : 'CorrectIllumination_Calculate',
                        'correctillumination_apply' : 'CorrectIllumination_Apply',
                        'createbatchfiles' : 'CreateBatchFiles',
                        'crop' : 'Crop',
                        'definegrid' : 'DefineGrid',
                        'enhanceorsuppressspeckles' : 'EnhanceOrSuppressSpeckles',
                        'expandorshrink' : 'ExpandOrShrink',
                        'exporttodatabase' : 'ExportToDatabase',
                        'exporttoexcel' : 'ExportToExcel',
                        'filterbyobjectmeasurement' : 'FilterByObjectMeasurement',
                        'findedges' : 'FindEdges',
                        'flagimage' : 'FlagImage',
                        'flipandrotate' : 'FlipAndRotate',
                        'graytocolor' : 'GrayToColor',
                        'identifyprimautomatic' : 'IdentifyPrimAutomatic',
                        'identifysecondary' : 'IdentifySecondary',
                        'identifytertiarysubregion' : 'IdentifyTertiarySubregion',
                        'imagemath' : 'ImageMath',
                        'invertforprinting' : 'InvertForPrinting',
                        'loadimages' : 'LoadImages',
                        'loadimagesnew' : 'LoadImagesNew',
                        'loadsingleimage' : 'LoadSingleImage',
                        'loadtext' : 'LoadText',
                        'makeprojection' : 'MakeProjection',
                        'maskimage' : 'MaskImage',
                        'measurecorrelation' : 'MeasureCorrelation',
                        'measureimageareaoccupied' : 'MeasureImageAreaOccupied',
                        'measureimagegranularity' : 'MeasureImageGranularity',
                        'measureimageintensity' : 'MeasureImageIntensity',
                        'measureimagequality' : 'MeasureImageQuality',
                        'measureobjectintensity' : 'MeasureObjectIntensity',
                        'measureobjectareashape' : 'MeasureObjectAreaShape',
                        'measureobjectneighbors' : 'MeasureObjectNeighbors',
                        'measureobjectradialdistribution' : 'MeasureObjectRadialDistribution',
                        'measuretexture' : 'MeasureTexture',
                        'morph' : 'Morph',
                        'overlay_outlines' : 'OverlayOutlines',
                        'pausecellprofiler': 'PauseCellProfiler',
                        'relate' : 'Relate',
                        'rescaleintensity' : 'RescaleIntensity',
                        'resize' : 'Resize',
                        'saveimages' : 'SaveImages',
                        'speedupcellprofiler' : 'SpeedUpCellProfiler',
                        'smooth' : 'Smooth',
                        'trackobjects' : 'TrackObjects',
                        }

# CP-Matlab to CP-python module substitutions
substitutions = {'FlagImageForQC' : 'FlagImage',
                 'KeepLargestObject' : 'FilterByObjectMeasurement',
                 'MeasureRadialDistribution' : 'MeasureObjectRadialDistribution',
                 'SmoothOrEnhance' : 'Smooth',
                 }

all_modules = {}
pymodules = []
badmodules = []

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
    for mod, name in pymodule_to_cpmodule.items():
        try:
            m = __import__('cellprofiler.modules.' + mod, globals(), locals(), [name])
            assert not name in all_modules, "Module %s appears more than once in module list"%(name)
        except Exception, e:
            badmodules.append((mod, e))
            continue

        try:
            pymodules.append(m)
            all_modules[name] = m.__dict__[name]
            check_module(m.__dict__[name], name)
            # attempt to instantiate
            all_modules[name]()
        except Exception, e:
            badmodules.append((mod, e))
            if name in all_modules:
                del all_modules[name]
                del pymodules[-1]

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
    return all_modules[module_class]()

def get_module_names():
    return all_modules.keys()

def reload_modules():
    for m in pymodules:
        reload(m)
    fill_modules()
