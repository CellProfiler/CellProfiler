"""Modules - pipeline processing modules for CellProfiler

"""
__version__="$Revision$"
import identifyprimautomatic as cpm_ipa
import loadimages as cpm_li
import colortogray as cpm_ctg
from applythreshold import ApplyThreshold
from crop import Crop
from saveimages import SaveImages
from measureobjectintensity import MeasureObjectIntensity
from exporttodatabase import ExportToDatabase
from identifysecondary import IdentifySecondary
from identifytertiarysubregion import IdentifyTertiarySubregion

def get_module_classes():
    return [ApplyThreshold,
            cpm_ctg.ColorToGray,
            Crop,
            ExportToDatabase,
            cpm_ipa.IdentifyPrimAutomatic,
            IdentifySecondary,
            IdentifyTertiarySubregion,
            cpm_li.LoadImages,
            MeasureObjectIntensity,
            SaveImages ]

def get_module_substitutions():
    """Return a dictionary of matlab module names and replacement classes
    
    """
    return {"LoadImages":cpm_li.LoadImages,
            "IdentifyPrimAutomatic":cpm_ipa.IdentifyPrimAutomatic,
            "IdentifySecondary":IdentifySecondary,
            "IdentifyTertiarySubregion":IdentifyTertiarySubregion,
            "ColorToGray":cpm_ctg.ColorToGray,
            "ApplyThreshold": ApplyThreshold,
            "SaveImages": SaveImages,
            "MeasureObjectIntensity": MeasureObjectIntensity,
            "ExportToDatabase": ExportToDatabase,
            "Crop": Crop
            }
    

 
