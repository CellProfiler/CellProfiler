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

import java.util.List;


/**
 * @author Lee Kamentsky
 * 
 * An ImageSetError which reports that more than one ipd has the same key
 *
 */
public class ImageSetDuplicateError extends ImageSetError {
	final private List<ImagePlaneDetailsStack> ipds;
	public ImageSetDuplicateError(
			String channelName, String message, List<String> key, 
			List<ImagePlaneDetailsStack> ipds) {
		super(channelName, message, key);
		this.ipds = ipds;
	}
	public List<ImagePlaneDetailsStack> getImagePlaneDetailsStacks() { return ipds; }
}
