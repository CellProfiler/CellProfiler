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
package org.cellprofiler.imageset.filter;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * @author Lee Kamentsky
 *
 * This predicate determines whether the candidate is an extension of a .tif file.
 */
public class IsTifPredicate extends AbstractTerminalPredicate<String> {
	final static public String SYMBOL = "istif";
	final static private Set<String> tifExtensions = 
		Collections.unmodifiableSet(
				new HashSet<String>(Arrays.asList("tif", "tiff", "ome.tif", "ome.tiff")));
	public IsTifPredicate() {
		super(String.class);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(String candidate) {
		return tifExtensions.contains(candidate.toLowerCase());
	}

}
