"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision: 8876 $"

import wx
import os.path
import glob

for f in glob.glob(os.path.join(__path__[0], "*.png")):
    globals()[os.path.basename(f)[:-4]] = wx.ImageFromStream(open(f))
