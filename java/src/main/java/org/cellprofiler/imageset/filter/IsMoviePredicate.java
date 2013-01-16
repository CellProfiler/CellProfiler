/**
 * 
 */
package org.cellprofiler.imageset.filter;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * @author Lee Kamentsky
 *
 * Is the extension a movie file extension?
 */
public class IsMoviePredicate extends AbstractTerminalPredicate<String> {
	final static public String SYMBOL = "ismovie";
	final static private Set<String> movieExtensions = 
		Collections.unmodifiableSet(
				new HashSet<String>(Arrays.asList("mov", "avi")));
	public IsMoviePredicate() {
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
		return movieExtensions.contains(candidate.toLowerCase());
	}
}
