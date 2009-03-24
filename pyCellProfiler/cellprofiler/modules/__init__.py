"""Modules - pipeline processing modules for CellProfiler

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import identifyprimautomatic as cpm_ipa
import loadimages as cpm_li
import colortogray as cpm_ctg
from applythreshold import ApplyThreshold
from correctillumination_calculate import CorrectIllumination_Calculate
from correctillumination_apply import CorrectIllumination_Apply
from crop import Crop
from exporttodatabase import ExportToDatabase
from graytocolor import GrayToColor
from identifysecondary import IdentifySecondary
from identifytertiarysubregion import IdentifyTertiarySubregion
from measureobjectintensity import MeasureObjectIntensity
from measureobjectareashape import MeasureObjectAreaShape
from saveimages import SaveImages

def get_module_classes():
    return [ApplyThreshold,
            cpm_ctg.ColorToGray,
            CorrectIllumination_Calculate,
            CorrectIllumination_Apply,
            Crop,
            ExportToDatabase,
            GrayToColor,
            cpm_ipa.IdentifyPrimAutomatic,
            IdentifySecondary,
            IdentifyTertiarySubregion,
            cpm_li.LoadImages,
            MeasureObjectAreaShape,
            MeasureObjectIntensity,
            SaveImages ]

def get_module_substitutions():
    """Return a dictionary of matlab module names and replacement classes
    
    """
    return {"ApplyThreshold": ApplyThreshold,
            "LoadImages":cpm_li.LoadImages,
            "ColorToGray":cpm_ctg.ColorToGray,
            "CorrectIllumination_Calculate":CorrectIllumination_Calculate,
            "CorrectIllumination_Apply":CorrectIllumination_Apply,
            "Crop": Crop,
            "ExportToDatabase": ExportToDatabase,
            "GrayToColor":GrayToColor,
            "IdentifyPrimAutomatic":cpm_ipa.IdentifyPrimAutomatic,
            "IdentifySecondary":IdentifySecondary,
            "IdentifyTertiarySubregion":IdentifyTertiarySubregion,
            "MeasureObjectAreaShape": MeasureObjectAreaShape,
            "MeasureObjectIntensity": MeasureObjectIntensity,
            "SaveImages": SaveImages
            }
    

 
