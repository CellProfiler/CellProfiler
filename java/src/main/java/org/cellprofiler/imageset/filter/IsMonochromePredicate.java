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

import net.imglib2.meta.Axes;

import org.cellprofiler.imageset.ImagePlane;
import org.cellprofiler.imageset.OMEMetadataExtractor;

/**
 * @author Lee Kamentsky
 * 
 * A predicate that determines whether an image plane
 * is a color image.
 *
 */
public class IsMonochromePredicate extends
		AbstractTerminalPredicate<ImagePlaneDetailsStack> {
	final static public String SYMBOL="ismonochrome";

	protected IsMonochromePredicate() {
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
		// We have a color image if it contains channels
		for (int i=0;i<candidate.numDimensions();i++) {
			if (candidate.axis(i).equals(Axes.CHANNEL)) {
				if (candidate.size(i) > 1) return false;
				for (ImagePlaneDetails ipd:candidate){
					if (ipd.imagePlane.getChannel() == ImagePlane.INTERLEAVED) return false;
				}
				return true;
			}
		}
		return false;
	}

}
