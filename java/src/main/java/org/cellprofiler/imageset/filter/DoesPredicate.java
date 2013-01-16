/**
 * 
 */
package org.cellprofiler.imageset.filter;

/**
 * The "Does" predicate is a pass-through. It returns the result of its
 * single subpredicate.
 * 
 * @author Lee Kamentsky
 *
 */
public class DoesPredicate<TINOUT> extends LogicPredicate<TINOUT> {
	public static final String SYMBOL = "does";

	/**
	 * The constructor takes the class of the input/output candidate
	 * @param klass
	 */
	public DoesPredicate(Class<TINOUT> klass) {
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
		return results[0];
	}

}
