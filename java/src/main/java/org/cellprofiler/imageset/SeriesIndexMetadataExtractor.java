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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
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
	static final public String SERIES_TAG = "Series";
	static final public String INDEX_TAG = "Frame";
	static final private String ZERO = "0";
	static final private List<String> metadataKeys = 
		Collections.unmodifiableList(Arrays.asList(SERIES_TAG, INDEX_TAG));
	/**
	 * Construct an extractor of the image plane series and index.
	 * 
	 */
	public SeriesIndexMetadataExtractor() {
			
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(ImagePlane source) {
		Map<String, String> result = new HashMap<String, String>(2);
		int series = source.getSeries().getSeries();
		int index = source.getIndex();
		result.put(SERIES_TAG, (series == 0)? ZERO: StringCache.intern(Integer.toString(series)));
		result.put(INDEX_TAG, (index == 0)? ZERO: StringCache.intern(Integer.toString(index)));
		return result;
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#getMetadataKeys()
	 */
	public List<String> getMetadataKeys() {
		return metadataKeys;
	}
	
}
