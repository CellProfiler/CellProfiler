"""Modules - pipeline processing modules for CellProfiler

"""
__version="$Revision: 1 $"

import identifyprimautomatic
import loadimages

module_classes = [loadimages.LoadImages,identifyprimautomatic.IdentifyPrimAutomatic]