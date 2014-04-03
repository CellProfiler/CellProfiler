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
package org.cellprofiler.imageset;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


/**
 * @author Lee Kamentsky
 *
 * An ImageSet is a collection of image stacks coallated for
 * processing during a CellProfiler cycle.
 */
public class ImageSet extends ArrayList<ImagePlaneDetailsStack> {
	private static final long serialVersionUID = -6824821112413090930L;
	final private List<String> key;
	/**
	 * Construct the image set from its image plane descriptors and key
	 * @param ipds
	 * @param key
	 */
	public ImageSet(Collection<ImagePlaneDetailsStack> ipds, List<String> key) {
		super(ipds);
		this.key = key;
	}
	
	public List<String> getKey() {
		return key;
	}
	
}
