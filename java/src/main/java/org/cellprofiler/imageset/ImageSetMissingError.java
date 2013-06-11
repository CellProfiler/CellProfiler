/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2013 Broad Institute
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
 * An ImageSetError reporting that the channel does not
 * have a matching IPD.
 */
public class ImageSetMissingError extends ImageSetError {

	public ImageSetMissingError(String channelName, String message,
			List<String> key) {
		super(channelName, message, key);
	}

}
