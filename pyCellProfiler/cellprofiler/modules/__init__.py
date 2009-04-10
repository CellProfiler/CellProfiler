"""Modules - pipeline processing modules for CellProfiler

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
from applythreshold import ApplyThreshold
from colortogray import ColorToGray
from correctillumination_calculate import CorrectIllumination_Calculate
from correctillumination_apply import CorrectIllumination_Apply
from crop import Crop
from exporttodatabase import ExportToDatabase
from exporttoexcel import ExportToExcel
from graytocolor import GrayToColor
from identifyprimautomatic import IdentifyPrimAutomatic
from identifysecondary import IdentifySecondary
from identifytertiarysubregion import IdentifyTertiarySubregion
from loadimages import LoadImages
from loadsingleimage import LoadSingleImage
from maskimage import MaskImage
from measureimageareaoccupied import MeasureImageAreaOccupied
from measureobjectintensity import MeasureObjectIntensity
from measureobjectareashape import MeasureObjectAreaShape
from measuretexture import MeasureTexture
from saveimages import SaveImages

def get_module_classes():
    return [ApplyThreshold,
            ColorToGray,
            CorrectIllumination_Calculate,
            CorrectIllumination_Apply,
            Crop,
            ExportToDatabase,
            ExportToExcel,
            GrayToColor,
            IdentifyPrimAutomatic,
            IdentifySecondary,
            IdentifyTertiarySubregion,
            LoadImages,
            LoadSingleImage,
            MaskImage,
            MeasureImageAreaOccupied,
            MeasureObjectAreaShape,
            MeasureObjectIntensity,
            MeasureTexture,
            SaveImages ]

def get_module_substitutions():
    """Return a dictionary of matlab module names and replacement classes
    
    """
    return {"ApplyThreshold": ApplyThreshold,
            "ColorToGray":ColorToGray,
            "CorrectIllumination_Calculate":CorrectIllumination_Calculate,
            "CorrectIllumination_Apply":CorrectIllumination_Apply,
            "Crop": Crop,
            "ExportToDatabase": ExportToDatabase,
            "ExportToExcel": ExportToExcel,
            "GrayToColor":GrayToColor,
            "IdentifyPrimAutomatic":IdentifyPrimAutomatic,
            "IdentifySecondary":IdentifySecondary,
            "IdentifyTertiarySubregion":IdentifyTertiarySubregion,
            "LoadImages":LoadImages,
            "LoadSingleImage":LoadSingleImage,
            "MaskImage": MaskImage,
            "MeasureImageAreaOccupied": MeasureImageAreaOccupied,
            "MeasureObjectAreaShape": MeasureObjectAreaShape,
            "MeasureObjectIntensity": MeasureObjectIntensity,
            "MeasureTexture": MeasureTexture,
            "SaveImages": SaveImages
            }
    

 
