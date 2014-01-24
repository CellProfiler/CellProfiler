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

import java.util.regex.Pattern;

import org.cellprofiler.imageset.MetadataUtils;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

/**
 * @author Lee Kamentsky
 *
 */
public class ContainsRegexpPredicate extends AbstractStringPredicate {
	final static public String SYMBOL = "containregexp";
	
	private Pattern pattern;
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}
	
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.AbstractStringPredicate#setLiteral(java.lang.String)
	 */
	@Override
	public void setLiteral(String literal) throws BadFilterExpressionException {
		pattern = MetadataUtils.compilePythonRegexp(literal, null);
		super.setLiteral(literal);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.AbstractStringPredicate#eval(java.lang.String, java.lang.String)
	 */
	@Override
	protected boolean eval(String candidate, String literal) {
		return pattern.matcher(candidate).find();
	}

}
