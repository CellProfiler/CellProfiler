package org.cellprofiler.imageset.filter;


import org.cellprofiler.imageset.ImageFile;

/**
 * @author Leek
 *
 */
public abstract class AbstractURLPredicate extends AbstractURLPredicateBase {
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(ImagePlaneDetails candidate) {
		String value = getValue(candidate.imagePlane.getImageFile());
		if (value == null) return false;
		return subpredicate.eval(value);
	}

	protected abstract String getValue(ImageFile candidate);

}
