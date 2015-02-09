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
 * The AndPredicate returns true iff all of its subpredicates evaluate
 * to true.
 * 
 * @author Lee Kamentsky
 * 
 *
 */
public class AndPredicate<TINOUT> extends LogicPredicate<TINOUT> {
	final static public String SYMBOL="and";
	public AndPredicate(Class<TINOUT> klass) {
		super(klass);
	}

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
		for (boolean result:results) {
			if (! result) return false;
		}
		return true;
	}

}
