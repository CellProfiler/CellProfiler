"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
#
# PIL decoder for the Cellomics DIB image format.
#

import struct
import PIL.Image as Image
import PIL.ImageFile as ImageFile

def uint_le(bytes):
    if len(bytes) == struct.calcsize("I"):
        return struct.unpack("<I", bytes)[0]
    elif len(bytes) == struct.calcsize("H"):
        return struct.unpack("<H", bytes)[0]
    else:
        raise ValueError, "Unexpected length: %d"%(len(bytes))

def _accept(prefix):
    return uint_le(prefix[0:4]) == 40


class DibImageFile(ImageFile.ImageFile):
    format = "DIB"
    format_description = "Cellomics DIB image"

    def _open(self):
        header = self.fp.read(52)
        header_length = uint_le(header[0:4])
        if header_length != 40:
            raise SyntaxError, "Cannot open DIB files with header length " \
                "not equal to 40."
        width = uint_le(header[4:8])
        height = uint_le(header[8:12])
        nchannels = uint_le(header[12:14])
        bit_depth = uint_le(header[14:16])
        compression = uint_le(header[16:20])
	# We have never seen a DIB file with more than one channel.
        # It seems reasonable to assume that the second channel would
        # follow the first, but this needs to be verified.
        self.size = width, height
        self.mode = "I"
        rawmode = "I;%d"%(bit_depth,)
        self.tile = [("raw", 
                      (0, 0, width, height),
                      self.fp.tell(),
                      (rawmode, 0, 1))]

Image.register_open("DIB", DibImageFile, _accept)
#Image.register_save("DIB", _save)

Image.register_extension("DIB", ".dib")
