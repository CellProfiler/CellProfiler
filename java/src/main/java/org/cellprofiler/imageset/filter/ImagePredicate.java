/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
 * All rights reserved.
 *
 * Please see the AUTHORS file for credits.
 *
 * Website: http://www.cellprofiler.org
 *
 */
package org.cellprofiler.imageset.filter;

import java.util.List;

import org.cellprofiler.imageset.ImagePlaneDetailsStack;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

/**
 * @author Lee Kamentsky
 * 
 * The ImagePredicate handles the specialized image-type determinations
 * such as color / grayscale images and stack frame / stack.
 * 
 * These require a bit more heuristic processing than the metadata
 * categories and are also grouped separately because of the UI.
 *
 */
public class ImagePredicate 
		implements FilterPredicate<ImagePlaneDetailsStack, ImagePlaneDetailsStack> {
	final static public String SYMBOL="image";
	private FilterPredicate<ImagePlaneDetailsStack, ?> subpredicate;

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setSubpredicates(java.util.List)
	 */
	public void setSubpredicates(
			List<FilterPredicate<ImagePlaneDetailsStack, ?>> subpredicates)
			throws BadFilterExpressionException {
		if (subpredicates.size() != 1)
			throw new BadFilterExpressionException("ImagePredicates have a single subpredicate");
		subpredicate = subpredicates.get(0);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setLiteral(java.lang.String)
	 */
	public void setLiteral(String literal) throws BadFilterExpressionException {
		throw new BadFilterExpressionException("Image predicates have subpredicates, not literals");
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(ImagePlaneDetailsStack candidate) {
		return subpredicate.eval(candidate);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getInputClass()
	 */
	public Class<ImagePlaneDetailsStack> getInputClass() {
		return ImagePlaneDetailsStack.class;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getOutputClass()
	 */
	public Class<ImagePlaneDetailsStack> getOutputClass() {
		return ImagePlaneDetailsStack.class;
	}

}
