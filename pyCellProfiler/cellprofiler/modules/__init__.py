"""Modules - pipeline processing modules for CellProfiler

"""
__version="$Revision$"

import identifyprimautomatic
import platonicmodule
import loadimages

module_classes = [platonicmodule.LoadImages,identifyprimautomatic.IdentifyPrimAutomatic]