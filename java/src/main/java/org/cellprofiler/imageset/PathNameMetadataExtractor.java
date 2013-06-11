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

/**
 * @author Lee Kamentsky
 *
 * This extractor router passes the path name to its children.
 * 
 */
public class PathNameMetadataExtractor extends
		MetadataExtractorAdapter<ImageFile, String> {

	protected PathNameMetadataExtractor(MetadataExtractor<String> subExtractor) {
		super(subExtractor);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractorAdapter#get(java.lang.Object)
	 */
	@Override
	protected String get(ImageFile source) {
		return source.getPathName();
	}

}
