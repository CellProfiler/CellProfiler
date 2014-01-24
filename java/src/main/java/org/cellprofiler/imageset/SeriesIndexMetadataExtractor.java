/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2014 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */
package org.cellprofiler.imageset;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Lee Kamentsky
 *
 * The ImagePlaneMetadataExtractor adds the image plane's series
 * and / or index to the image's metadata.
 * 
 * This information is crucial for image stacks, but otherwise
 * superfluous, so it should be conditionally added to the filters.
 */
public class SeriesIndexMetadataExtractor implements MetadataExtractor<ImagePlane> {
	final private String seriesFormat;
	final private String indexFormat;
	static final public String SERIES_TAG = "Series";
	static final public String INDEX_TAG = "Frame";
	/**
	 * Construct an extractor of the image plane series and index. We zero-pad
	 * the numbers so they sort both alphabetically and numerically
	 * 
	 * @param seriesDigits number of digits to use to display series. Zero means
	 *        do not record series information.
	 * @param indexDigits number of digits to use to display index. Zero means
	 *        do not record index information.
	 */
	public SeriesIndexMetadataExtractor(int seriesDigits, int indexDigits) {
		if (seriesDigits == 0) {
			seriesFormat = null;
		} else {
			seriesFormat = String.format("%%0%dd", seriesDigits);
		}
		if (indexDigits == 0) {
			indexFormat = null;
		} else {
			indexFormat = String.format("%%0%dd", indexDigits);
		}
			
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(ImagePlane source) {
		final int nFormats = ((seriesFormat == null)?0:1) + ((indexFormat==null)?0:1);
		Map<String, String> result = new HashMap<String, String>(nFormats);
		if (seriesFormat != null) {
			result.put(SERIES_TAG, String.format(seriesFormat, source.getSeries()));
		}
		if (indexFormat != null) {
			result.put(INDEX_TAG, String.format(indexFormat, source.getIndex()));
		}
		return result;
	}
	
}
