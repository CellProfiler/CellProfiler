/**
 * 
 */
package org.cellprofiler.imageset.filter;

import org.cellprofiler.imageset.ImageFile;


/**
 * @author Lee Kamentsky
 *
 */
public class FileNamePredicate extends AbstractURLPredicate {
	final static public String SYMBOL = "file";
	public FileNamePredicate() {
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.AbstractURLPredicate#getValue(org.cellprofiler.imageset.filter.ImageFile)
	 */
	@Override
	protected String getValue(ImageFile candidate) {
		return candidate.getFileName();
	}

}
