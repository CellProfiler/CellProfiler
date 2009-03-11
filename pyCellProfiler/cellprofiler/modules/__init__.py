"""Modules - pipeline processing modules for CellProfiler

"""
__version__="$Revision$"
import identifyprimautomatic as cpm_ipa
import loadimages as cpm_li
import colortogray as cpm_ctg
from applythreshold import ApplyThreshold
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
    

 
