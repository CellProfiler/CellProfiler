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
 * A filter predicate can implement the TokenParser interface if it
 * wants to be in charge of generating its own subpredicates.
 */
public interface TokenParser<TIN, TOUT> extends FilterPredicate<TIN, TOUT> {
	/**
	 * Parse a token, returning a filter predicate representing the token
	 *  
	 * @param token a token parsed out of the filter expression
	 * @return the filter subpredicate
	 */
	public FilterPredicate<TOUT, ?> parse(String token);

}
