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

package org.cellprofiler.imageset.filter;

import java.util.List;

import org.cellprofiler.imageset.ImageFile;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

/**
 * @author Lee Kamentsky
 *
 * A class for adapting filters of one type to another
 * 
 * @param <TIN> The type of the input.
 * @param <TINTERMEDIATE>
 * @param <TOUT>
 */
abstract public class FilterAdapter<TIN, TINTERMEDIATE, TOUT>
    implements FilterPredicate<TIN, TOUT>{

	protected FilterPredicate<TINTERMEDIATE, TOUT> predicate;

	/**
	 * Constructor to handle the proper parameterization of types
	 * 
	 * @param cIn
	 * @param cIntermediate
	 * @param cOut
	 */
	protected FilterAdapter(Class<TIN> cIn, Class<TINTERMEDIATE> cIntermediate, Class<TOUT> cOut) {
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return predicate.getSymbol();
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
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getOutputClass()
	 */
	public Class<TOUT> getOutputClass() {
		return predicate.getOutputClass();
	}
	
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(TIN candidate) {
		return predicate.eval(getValue(candidate)); 
	}
	
	/**
	 * Get the value for the adaptation (e.g. the ImageFile for an IPD)
	 * to be passed onto the controlled filter
	 * 
	 * @param candidate the candidate being filtered
	 * @return the intermediate value
	 */
	abstract public TINTERMEDIATE getValue(TIN candidate);

}