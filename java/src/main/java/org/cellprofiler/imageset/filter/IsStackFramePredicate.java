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

import org.cellprofiler.imageset.OMEMetadataExtractor;

/**
 * @author Lee Kamentsky
 * 
 * A predicate that determines whether an image plane stack
 * is part of an image stack.
 * 
 * An image "plane" is a stack if it has M_T or M_Z
 * metadata.
 */
public class IsStackFramePredicate extends
		AbstractTerminalPredicate<ImagePlaneDetails> {
	final static public String SYMBOL="isstackframe";

	protected IsStackFramePredicate() {
		super(ImagePlaneDetails.class);
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
	public boolean eval(ImagePlaneDetails candidate) {
		for (final String key: new String [] {OMEMetadataExtractor.MD_T, OMEMetadataExtractor.MD_Z} ) {
			if ( candidate.metadata.containsKey(key)) return true;
		}
		return false;
	}
}
