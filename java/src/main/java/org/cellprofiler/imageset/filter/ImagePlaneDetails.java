/**
 * 
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
