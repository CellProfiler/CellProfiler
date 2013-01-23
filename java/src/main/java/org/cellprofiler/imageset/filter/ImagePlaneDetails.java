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
package org.cellprofiler.imageset.filter;

import java.util.Map;

import org.cellprofiler.imageset.ImagePlane;

/**
 * @author Lee Kamentsky
 *
 * The ImagePlaneDetails class couples an ImagePlane to the
 * metadata extracted from the ImagePlane by the extractors.
 */
public class ImagePlaneDetails {
	/**
	 * Metadata key/value pairs collected on the image plane
	 */
	public final Map<String, String> metadata;
	
	/**
	 * The image plane to be filtered 
	 */
	public final ImagePlane imagePlane;
	public ImagePlaneDetails(ImagePlane imagePlane, Map<String, String> metadata) {
		this.imagePlane = imagePlane;
		this.metadata = metadata;
	}
}
