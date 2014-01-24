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

import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

/**
 * @author Lee Kamentsky
 *
 * A filter predicate determines whether an image plane is kept
 * or filtered out.
 * 
 * @param <TIN> the type that the filter evaluates.
 * @param <TOUT> the type required of its subpredicates.
 */
public interface FilterPredicate<TIN, TOUT> {
	/**
	 * The symbol is the string that represents this predicate
	 * in the filter's serialization
	 * 
	 * @return the symbol that represents this predicate.
	 */
	public String getSymbol();
	
	/**
	 * Subpredicates are context-dependent filter predicates. For instance,
	 * an "And" filter predicate might take any number of subpredicates
	 * each of which are evaluated. The predicate would then return
	 * the boolean union of all of the subpredicate evaluations. A "Filename"
	 * predicate might parse the file name from the URL and pass that to a single
	 * subpredicate for evaluation.
	 * 
	 * @param subpredicates
	 * @throws BadFilterExpressionException TODO
	 */
	public void setSubpredicates(List<FilterPredicate<TOUT, ?>> subpredicates) throws BadFilterExpressionException;
	
	/**
	 * Set a literal value that supplies the qualifications for the target
	 * @param literal the literal value to be consumed.
	 * @throws BadFilterExpressionException TODO
	 */
	public void setLiteral(String literal) throws BadFilterExpressionException;
	
	/**
	 * Evaluate the candidate to see whether it passes the filter
	 * 
	 * @param candidate - the entity to be checked, such as a string or ImagePlane
	 * @return true if the candidate should be kept, false if it should be filtered out
	 */
	public boolean eval(TIN candidate);
	
	/**
	 * @return the base class for acceptable inputs
	 */
	public Class<TIN> getInputClass();
	
	/**
	 * @return the base class for acceptable outputs. return a base class of null
	 *         if the predicate terminates with a literal instead of subpredicates
	 *         or if it is an endpoint in and of itself (e.g. extension is image extension).
	 */
	public Class<TOUT> getOutputClass();
}
