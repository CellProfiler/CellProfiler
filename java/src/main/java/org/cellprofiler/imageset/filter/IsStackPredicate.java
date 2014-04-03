/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2014 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */
package org.cellprofiler.imageset.filter;

import net.imglib2.Axis;
import net.imglib2.meta.Axes;

import org.cellprofiler.imageset.ImagePlaneDetailsStack;
import org.cellprofiler.imageset.OMEPlaneMetadataExtractor;

/**
 * @author Lee Kamentsky
 * 
 * A predicate that determines whether a stack
 * has a single frame or has multiple ones
 * 
 */
public class IsStackPredicate extends
		AbstractTerminalPredicate<ImagePlaneDetailsStack> {
	final static public String SYMBOL="isstack";

	protected IsStackPredicate() {
		super(ImagePlaneDetailsStack.class);
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
	public boolean eval(ImagePlaneDetailsStack candidate) {
		for (int i=0; i<candidate.numDimensions(); i++) {
			if (candidate.size(i) > 1) return true;
		}
		return false;
	}
}
