/**
 * 
 */
package org.cellprofiler.imageset.filter;

/**
 * @author Lee Kamentsky
 *
 * Is the extension a Flex file extension?
 */
public class IsFlexPredicate extends AbstractTerminalPredicate<String> {
	public IsFlexPredicate() {
		super(String.class);
	}

	final static public String SYMBOL = "isflex";
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(String candidate) {
		return candidate.toLowerCase().equals("flex");
	}

}
