/**
 * 
 */
package org.cellprofiler.imageset.filter;

/**
 * @author Lee Kamentsky
 * 
 * A predicate that determines whether the candidate extension is that of a .PNG file.
 *
 */
public class IsPNGPredicate extends AbstractTerminalPredicate<String> {
	final static public String SYMBOL = "ispng";
	public IsPNGPredicate() {
		super(String.class);
	}

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
		return candidate.toLowerCase().equals("png");
	}
}
