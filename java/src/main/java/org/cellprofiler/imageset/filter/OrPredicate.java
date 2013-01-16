/**
 * 
 */
package org.cellprofiler.imageset.filter;

/**
 * @author Lee Kamentsky
 *
 */
public class OrPredicate<TINOUT> extends LogicPredicate<TINOUT> {
	final static public String SYMBOL = "or"; 
	/**
	 * Construct the OrPredicate by specifying the class of its candidate
	 * @param klass the class of the candidate for this predicate and
	 *        its subpredicates.
	 */
	public OrPredicate(Class<TINOUT> klass) {
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
			if (result) return true;
		}
		return false;
	}

}
