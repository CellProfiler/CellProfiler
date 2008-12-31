"""Modules - pipeline processing modules for CellProfiler

"""
__version__="$Revision$"
import identifyprimautomatic as cpm_ipa
import platonicmodule as cpm_pm
import loadimages as cpm_li

def get_module_classes():
    return [cpm_pm.LoadImages,
            cpm_ipa.IdentifyPrimAutomatic]

def get_module_substitutions():
    """Return a dictionary of matlab module names and replacement classes
    
    """
    return {"LoadImages":cpm_pm.LoadImages,
            "IdentifyPrimAutomatic":cpm_ipa.IdentifyPrimAutomatic
            }
    

 