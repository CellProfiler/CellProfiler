/**
 * 
 */
package org.cellprofiler.imageset.filter;

/**
 * The AndPredicate returns true iff all of its subpredicates evaluate
 * to true.
 * 
 * @author Lee Kamentsky
 * 
 *
 */
public class AndPredicate<TINOUT> extends LogicPredicate<TINOUT> {
	final static public String SYMBOL="and";
	public AndPredicate(Class<TINOUT> klass) {
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
			if (! result) return false;
		}
		return true;
	}

}
