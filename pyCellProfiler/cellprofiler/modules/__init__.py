"""Modules - pipeline processing modules for CellProfiler

"""
__version="$Revision$"

import identifyprimautomatic
import loadimages

module_classes = [loadimages.LoadImages,identifyprimautomatic.IdentifyPrimAutomatic]