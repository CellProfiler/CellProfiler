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
import java.util.List;
import java.util.Map;


/**
 * @author Lee Kamentsky
 *
 * CellProfiler will make up a well name if it sees WellRow and WellColumn
 * in the metadata. You can tack this metadata extractor onto the
 * end of your chain to get the same effect. The rules are:
 * 
 * If the metadata contains a "Well" key (case-insensitive matching), don't process further.
 * Row metadata items are "wellrow", "well_row" and "row".
 * Column metadata items are "wellcol", "well_col", "wellcolumn", "well_column", "column", and "col"
 * If the metadata contains one of each (case-insensitive matching), combine to make a "Well"
 * metadata item.
 */
public class WellMetadataExtractor implements
		MetadataExtractor<ImagePlaneDetails> {
	final static public String WELL = "Well";
	final static private List<String> rowKeys = Arrays.asList("wellrow", "well_row", "row");
	final static private List<String> columnKeys = Arrays.asList("wellcol", "well_col", "wellcolumn", "well_column", "column", "col");
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(ImagePlaneDetails source) {
		String wellRow = null;
		String wellColumn = null;
		for (String key:source) {
			final String lcKey = key.toLowerCase();
			if (lcKey.equals("well")) return emptyMap;
			if (rowKeys.contains(lcKey)) {
				wellRow = source.get(key);
			} else if (columnKeys.contains(lcKey)) {
				wellColumn = source.get(key);
			}
		}
		if ((wellRow != null) && (wellColumn != null)) {
			return Collections.singletonMap(WELL, StringCache.intern(wellRow + wellColumn));
		}
		return emptyMap;
	}

}
