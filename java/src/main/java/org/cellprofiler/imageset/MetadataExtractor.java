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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Lee Kamentsky
 * 
 * Classes that implement MetadataExtractor can extract metadata
 * from something.
 *
 * @param <T> Extract metadata from these sorts of things.
 *            An image file metadata extractor might extract
 *            metadata from the image file's OME metadata or
 *            might house metadata extractors for the file
 *            name and path. An image plane metadata extractor
 *            might extract plane-specific metadata or might
 *            house the image file metadata extractor.
 */
public interface MetadataExtractor<T> {
	final static public Map<String, String> emptyMap = Collections.unmodifiableMap(
			new HashMap<String, String>());
	/**
	 * Extract metadata from a source
	 * 
	 * @param source - the source of the metadata
	 * @return a key / value map of metadata entries
	 */
	public Map<String, String> extract(T source);

}
