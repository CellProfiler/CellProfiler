/**
 * 
 */
package org.cellprofiler.imageset.filter;

import java.util.Arrays;
import java.util.List;

/**
 * @author Lee Kamentsky
 *
 */
public class IsImagePredicate extends AbstractTerminalPredicate<String> {
	final static public String SYMBOL = "isimage";
	
	@SuppressWarnings("unchecked")
	static List<AbstractTerminalPredicate<String>> predicates = Arrays.asList(
		new IsTifPredicate(),
		new IsJPEGPredicate(),
		new IsPNGPredicate());
	
	public IsImagePredicate() {
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
		for (AbstractTerminalPredicate<String> predicate:predicates) {
			if (predicate.eval(candidate)) return true;
		}
		return false;
	}

}
