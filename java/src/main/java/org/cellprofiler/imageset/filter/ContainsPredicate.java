/**
 * 
 */
package org.cellprofiler.imageset.filter;

/**
 * @author Lee Kamentsky
 *
 * A predicate that tests whether the candidate contains the literal.
 */
public class ContainsPredicate extends AbstractStringPredicate {
	final static public String SYMBOL = "contain";
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.AbstractStringPredicate#eval(java.lang.String, java.lang.String)
	 */
	@Override
	protected boolean eval(String candidate, String literal) {
		return candidate.indexOf(literal) >= 0;
	}

}
