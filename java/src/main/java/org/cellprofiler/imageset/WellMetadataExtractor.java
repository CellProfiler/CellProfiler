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
	final static private List<String> metadataKeys = Collections.singletonList(WELL);
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
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#getMetadataKeys()
	 */
	public List<String> getMetadataKeys() {
		return metadataKeys;
	}
	/**
	 * Determines whether you might need the WellMetadataExtractor
	 * as part of your extractor chain, based on the other
	 * keys in your extractor.
	 * 
	 * @param keys - the keys in your extractor.
	 * @return true if the WellMetadataExtractor might be able
	 * to construct a well name from your other metadata
	 */
	public static boolean maybeYouNeedThis(List<String> keys) {
		// Must have one of each of a row and column key
		top_loop:
		for (List<String> targetList:new List[] {rowKeys, columnKeys}) {
			for (String key:keys) {
				if (targetList.contains(key.toLowerCase())) continue top_loop;
			}
			return false;
		}
		return true;
	}

}
