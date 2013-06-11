/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2013 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */
package org.cellprofiler.imageset;

import java.util.HashMap;
import java.util.Map;

import ome.xml.model.Channel;
import ome.xml.model.Image;
import ome.xml.model.OME;
import ome.xml.model.Pixels;
import ome.xml.model.Plane;
import ome.xml.model.primitives.NonNegativeInteger;
import ome.xml.model.primitives.PositiveInteger;

/**
 * @author Lee Kamentsky
 * 
 * Extract metadata items from the OME metadata similarly
 * to the way it's done in Python CellProfiler.
 *
 */
public class OMEMetadataExtractor implements MetadataExtractor<ImagePlane> {
	final static public String MD_C = "C";
	final static public String MD_T = "T";
	final static public String MD_Z = "Z";
	final static public String MD_COLOR_FORMAT = "ColorFormat";
	final static public String MD_RGB = "RGB";
	final static public String MD_MONOCHROME = "monochrome";
	final static public String MD_PLANAR = "Planar";
	final static public String MD_CHANNEL_NAME = "ChannelName";

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(ImagePlane source) {
		ImageFile imageFile = source.getImageFile();
		OME ome = imageFile.getMetadata();
		if (ome == null) {
			return emptyMap;
		}
		if (ome.sizeOfImageList() <= source.getSeries()) 
			return emptyMap;
		HashMap<String, String> map = new HashMap<String, String>();
		Image imageMetadata = ome.getImage(source.getSeries());
		Pixels pixels = imageMetadata.getPixels();
		if (pixels.sizeOfPlaneList() > source.getIndex()) {
			Plane plane = pixels.getPlane(source.getIndex());
			putIfNotNull(map, MD_C, plane.getTheC().toString());
			putIfNotNull(map, MD_T, plane.getTheT().toString());
			putIfNotNull(map, MD_Z, plane.getTheZ().toString());
			final NonNegativeInteger c = plane.getTheC();
			if (c != null) {
				final int cidx = c.getValue().intValue();
				if (pixels.sizeOfChannelList() > cidx) {
					Channel channel = pixels.getChannel(cidx);
					if (channel != null) {
						putIfNotNull(map, MD_CHANNEL_NAME, channel.getName());
						final PositiveInteger samplesPerPixel = channel.getSamplesPerPixel();
						if (samplesPerPixel != null) {
							final int nSamplesPerPixel = samplesPerPixel.getValue().intValue();
							map.put(MD_COLOR_FORMAT, (nSamplesPerPixel == 1)?MD_MONOCHROME:MD_RGB);
						}
					}
				}
			}
		} else {
			if (pixels.getSizeC().getValue().intValue() == 1) {
				map.put(MD_COLOR_FORMAT, MD_MONOCHROME);
				Channel channel = pixels.getChannel(0);
				if (channel != null) {
					putIfNotNull(map, MD_CHANNEL_NAME, channel.getName());
				}
			} else if (pixels.sizeOfChannelList() == 1) {
				map.put(MD_COLOR_FORMAT, MD_RGB);
			} else {
				map.put(MD_COLOR_FORMAT, MD_MONOCHROME);
			}
			/*
			 * Assume it's a movie if there's no plane data and there is more than one frame. The index gives the T.
			 */
			if (pixels.getSizeT().getValue().intValue() > 1) 
				map.put(MD_T, StringCache.intern(Integer.toString(source.getIndex())));
			else if (pixels.getSizeZ().getValue().intValue() > 1)
				map.put(MD_Z, StringCache.intern(Integer.toString(source.getIndex())));
		}
		return map;
	}
	static private void putIfNotNull(Map<String, String> map, String key, String value) {
		if (value != null) map.put(key, StringCache.intern(value));
	}
}
