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
 * A predicate that terminates an expression (without taking a literal)
 * An example is the IsTifPredicate which determines if
 * the extension matches one of the .tif extensions.
 */
public abstract class AbstractTerminalPredicate<TIN> implements
		FilterPredicate<TIN, Object> {
	final private Class<TIN> klass;
	
	protected AbstractTerminalPredicate(Class<TIN> klass) {
		this.klass = klass;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setSubpredicates(java.util.List)
	 */
	public void setSubpredicates(List<FilterPredicate<Object, ?>> subpredicates)
			throws BadFilterExpressionException {
		throw new BadFilterExpressionException(
				String.format("The %s predicate does not take subpredicates.", getSymbol()));
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setLiteral(java.lang.String)
	 */
	public void setLiteral(String literal) throws BadFilterExpressionException {
		throw new BadFilterExpressionException(
				String.format("The %s predicate does not take a literal.", getSymbol()));
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getInputClass()
	 */
	public Class<TIN> getInputClass() {
		return klass;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getOutputClass()
	 */
	public Class<Object> getOutputClass() {
		return null;
	}
}
