/**
 * 
 */
package org.cellprofiler.imageset.filter;

/**
 * @author Lee Kamentsky
 *
 */
public class DoesNotPredicate<TINOUT> extends LogicPredicate<TINOUT> {
	public DoesNotPredicate(Class<TINOUT> klass) {
		super(klass);
	}

	final public static String SYMBOL="doesnot";

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
		return ! results[0];
	}

}
