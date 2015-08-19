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

import java.util.List;

import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

/**
 * @author Lee Kamentsky
 * 
 * A Filter predicate that inverts the logical sense of another
 * predicate.
 *
 */
public class FilterPredicateInverter<TIN, TOUT> implements
		FilterPredicate<TIN, TOUT> {

	final FilterPredicate<TIN, TOUT> predicate;
	final String symbol;
	/**
	 * Constructor - initialize the inverter with the inverse
	 * filter predicate and the parsing symbol to use.
	 * 
	 * @param predicate
	 * @param symbol
	 */
	protected FilterPredicateInverter(FilterPredicate<TIN, TOUT> predicate, String symbol){
		this.predicate = predicate;
		this.symbol = symbol;
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return symbol;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setSubpredicates(java.util.List)
	 */
	public void setSubpredicates(List<FilterPredicate<TOUT, ?>> subpredicates)
			throws BadFilterExpressionException {
		predicate.setSubpredicates(subpredicates);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setLiteral(java.lang.String)
	 */
	public void setLiteral(String literal) throws BadFilterExpressionException {
		predicate.setLiteral(literal);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(TIN candidate) {
		return ! predicate.eval(candidate);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getInputClass()
	 */
	public Class<TIN> getInputClass() {
		return predicate.getInputClass();
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getOutputClass()
	 */
	public Class<TOUT> getOutputClass() {
		return predicate.getOutputClass();
	}

}
