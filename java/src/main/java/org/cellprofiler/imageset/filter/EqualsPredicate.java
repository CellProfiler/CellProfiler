/**
 * 
 */
package org.cellprofiler.imageset.filter;

/**
 * @author Lee Kamentsky
 *
 */
public class EqualsPredicate extends AbstractStringPredicate {
	final static public String SYMBOL = "eq"; 
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
		return candidate.equals(literal);
	}

}
