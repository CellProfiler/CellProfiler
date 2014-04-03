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
 * Extract file-level items from an ImageFile's OME metadata.
 * 
 */
public class OMEFileMetadataExtractor implements MetadataExtractor<ImageFile> {

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(ImageFile source) {
		// Currently, we extract no file-level data
		// from the OME metadata.
		final Map<String, String> map = new HashMap<String, String>();
		return map;
	}

}
