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

import org.cellprofiler.imageset.ImagePlaneDetailsStack;
import org.cellprofiler.imageset.OMEPlaneMetadataExtractor;

/**
 * @author Lee Kamentsky
 * 
 * A predicate that determines whether an ImagePlaneDetailsStack
 * has just a single image plane.
 * 
 */
public class IsStackFramePredicate extends
		FilterPredicateInverter<ImagePlaneDetailsStack, Object> {
	final static public String SYMBOL="isstackframe";

	public IsStackFramePredicate() {
		super(new IsStackPredicate(), SYMBOL);
	}
}
