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


/**
 * @author Lee Kamentsky
 *
 * A metadata extractor router that routes the ImageFile's
 * file name to child extractors.
 */
public class FileNameMetadataExtractor extends
		MetadataExtractorAdapter<ImageFile, String> {

	public FileNameMetadataExtractor(MetadataExtractor<String> subExtractor) {
		super(subExtractor);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractorAdapter#get(java.lang.Object)
	 */
	@Override
	protected String get(ImageFile source) {
		return source.getFileName();
	}

}
