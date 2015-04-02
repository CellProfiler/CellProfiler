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
package org.cellprofiler.imageset.filter;

/**
 * @author Lee Kamentsky
 *
 */
public class DoesNotPredicate<TINOUT> extends LogicPredicate<TINOUT> {
	public DoesNotPredicate(Class<TINOUT> klass) {
		super(klass);
	}

	final public static String SYMBOL="doesnot";

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.LogicPredicate#eval(boolean[])
	 */
	@Override
	protected boolean eval(boolean[] results) {
		return ! results[0];
	}

}
