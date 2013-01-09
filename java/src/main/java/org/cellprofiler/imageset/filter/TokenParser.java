/**
 * 
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
