/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */

package org.cellprofiler.imageset;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ome.xml.model.Image;
import ome.xml.model.Pixels;
import ome.xml.model.Plate;
import ome.xml.model.Well;
import ome.xml.model.WellSample;

/**
 * @author Lee Kamentsky
 *
 * Apply OME metadata at the Pixels and Image level
 * to an ImageSeries
 */
public class OMESeriesMetadataExtractor implements
		MetadataExtractor<ImageSeries> {
	final static public String MD_SIZE_X = "SizeX";
	final static public String MD_SIZE_Y = "SizeY";
	final static public String MD_SIZE_C = "SizeC";
	final static public String MD_SIZE_T = "SizeT";
	final static public String MD_SIZE_Z = "SizeZ";
	final static public String MD_SITE = "Site";
	final static public String MD_WELL = "Well";
	final static public String MD_PLATE = "Plate";
	final static private List<String> metadataKeys =
		Collections.unmodifiableList(Arrays.asList(
				MD_SIZE_X, MD_SIZE_Y, MD_SIZE_C,
				MD_SIZE_T, MD_SIZE_Z, MD_SITE, MD_WELL, MD_PLATE));

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(ImageSeries source) {
		final Map<String, String> map = new HashMap<String, String>();
		Image image = source.getOMEImage();
		if (image != null) {
			Pixels pixels = image.getPixels();
			putIfNotNull(map, MD_SIZE_X, pixels.getSizeX());
			putIfNotNull(map, MD_SIZE_Y, pixels.getSizeY());
			putIfNotNull(map, MD_SIZE_C, pixels.getSizeC());
			putIfNotNull(map, MD_SIZE_T, pixels.getSizeT());
			putIfNotNull(map, MD_SIZE_Z, pixels.getSizeZ());
			if (image.sizeOfLinkedWellSampleList() == 1) {
				final WellSample wellSample = image.getLinkedWellSample(0);
				putIfNotNull(map, MD_SITE, wellSample.getIndex());
				final Well well = wellSample.getWell();
				if (well != null) {
					map.put(MD_WELL, getWellName(well.getRow().getValue(), well.getColumn().getValue()));
					Plate plate = well.getPlate();
					if (plate != null) {
						map.put(MD_PLATE, well.getPlate().getName());
					}
				}
			}
		} else {
			map.put(MD_SIZE_C, StringCache.intern("1"));
			map.put(MD_SIZE_T, StringCache.intern("1"));
			map.put(MD_SIZE_Z, StringCache.intern("1"));
		}
		return map;
	}	
	static private <T> void putIfNotNull(Map<String, String> map, String key, T value) {
		if (value != null) map.put(key, StringCache.intern(value.toString()));
	}
	static private String getWellName(int rowIdx, int colIdx) {
		String rowName = "";
		while (rowIdx >= 26) {
			rowIdx -= 26;
			rowName = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".charAt(rowIdx % 26) + rowName;
			rowIdx = rowIdx / 26;
		}
		rowName = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".charAt(rowIdx % 26) + rowName;
		return String.format("%s%02d", rowName, colIdx);
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#getMetadataKeys()
	 */
	public List<String> getMetadataKeys() {
		return metadataKeys;
	}
}
