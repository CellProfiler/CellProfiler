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

import java.util.Map;

/**
 * @author Lee Kamentsky
 * 
 * Adapts the input metadata source to a metadata
 * extractor for the output metadata extractor
 *
 */
public abstract class MetadataExtractorAdapter<TIN, TOUT> implements
		MetadataExtractor<TIN> {
	final private MetadataExtractor<TOUT> subExtractor;
	protected MetadataExtractorAdapter(MetadataExtractor<TOUT> subExtractor) {
		this.subExtractor = subExtractor;
	}

	/**
	 * Get the part of the source that's appropriate for the controlled extractor
	 * @param source
	 * 
	 * @return the source to pass to the controlled extractor
	 */
	abstract protected TOUT get(TIN source);
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(TIN source) {
		return subExtractor.extract(get(source));
	}
}
