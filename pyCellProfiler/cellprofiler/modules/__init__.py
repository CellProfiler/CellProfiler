"""Modules - pipeline processing modules for CellProfiler

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

# python modules and their corresponding cellprofiler.module classes
pymodule_to_cpmodule = {'align' : 'Align',
                        'applythreshold' : 'ApplyThreshold',
                        'calculatemath' : 'CalculateMath',
                        'colortogray' : 'ColorToGray',
                        'converttoimage' : 'ConvertToImage',
                        'correctillumination_calculate' : 'CorrectIllumination_Calculate',
                        'correctillumination_apply' : 'CorrectIllumination_Apply',
                        'createbatchfiles' : 'CreateBatchFiles',
                        'crop' : 'Crop',
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
def fill_modules():
    del pymodules[:]
    for mod, name in pymodule_to_cpmodule.items():
        print mod, name
        m = __import__('cellprofiler.modules.' + mod, globals(), locals(), [name])
        pymodules.append(m)
        all_modules[mod] = m.__dict__[name]
fill_modules()
    
__all__ = ['instantiate_module', 'get_module_classes', 'reload_modules']

def instantiate_module(module_name):
    if module_name in substitutions: 
        module_name = substitutions[module_name]
    return all_modules[module_name.split('.')[-1]]()

def get_module_names():
    return all_modules.keys()

def reload_modules():
    for m in pymodules:
        reload(m)
    fill_modules()
