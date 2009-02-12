"""Modules - pipeline processing modules for CellProfiler

"""
__version__="$Revision$"
import identifyprimautomatic as cpm_ipa
import loadimages as cpm_li
import colortogray as cpm_ctg
from applythreshold import ApplyThreshold
from saveimages import SaveImages
from measureobjectintensity import MeasureObjectIntensity

def get_module_classes():
    return [cpm_li.LoadImages,
            cpm_ipa.IdentifyPrimAutomatic,
            cpm_ctg.ColorToGray,
            ApplyThreshold,
            SaveImages,
            MeasureObjectIntensity]

def get_module_substitutions():
    """Return a dictionary of matlab module names and replacement classes
    
    """
    return {"LoadImages":cpm_li.LoadImages,
            "IdentifyPrimAutomatic":cpm_ipa.IdentifyPrimAutomatic,
            "ColorToGray":cpm_ctg.ColorToGray,
            "ApplyThreshold": ApplyThreshold,
            "SaveImages": SaveImages,
            "MeasureObjectIntensity": MeasureObjectIntensity
            }
    

 
