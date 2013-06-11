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
import java.util.List;

/**
 * @author Lee Kamentsky
 *
 */
public class IsImagePredicate extends AbstractTerminalPredicate<String> {
	final static public String SYMBOL = "isimage";
	
	@SuppressWarnings("unchecked")
	static List<AbstractTerminalPredicate<String>> predicates = Arrays.asList(
		new IsTifPredicate(),
		new IsJPEGPredicate(),
		new IsPNGPredicate());
	
	public IsImagePredicate() {
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
		for (AbstractTerminalPredicate<String> predicate:predicates) {
			if (predicate.eval(candidate)) return true;
		}
		return false;
	}

}
