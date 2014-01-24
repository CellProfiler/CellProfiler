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


/**
 * @author Lee Kamentsky
 *
 * A filter predicate that takes an ImagePlaneDetals as input
 * 
 * @param <TOUT>
 */
public abstract class AbstractImagePlaneDetailsPredicate<TOUT> 
	implements FilterPredicate<ImagePlaneDetails, TOUT> {

	public Class<ImagePlaneDetails> getInputClass() {
		return ImagePlaneDetails.class;
	}

}
