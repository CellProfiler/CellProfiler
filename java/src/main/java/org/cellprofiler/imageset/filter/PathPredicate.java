/**
 * 
 */
package org.cellprofiler.imageset.filter;

import org.cellprofiler.imageset.ImageFile;

/**
 * @author Lee Kamentsky
 *
 */
public class PathPredicate extends AbstractURLPredicate {
	final static public String SYMBOL = "directory";
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.AbstractURLPredicate#getValue(org.cellprofiler.imageset.ImageFile)
	 */
	@Override
	protected String getValue(ImageFile candidate) {
		return candidate.getPathName();
	}

}
