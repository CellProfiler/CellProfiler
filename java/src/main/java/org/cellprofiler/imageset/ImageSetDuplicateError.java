/**
 * 
 */
package org.cellprofiler.imageset;

import java.util.List;

import org.cellprofiler.imageset.filter.ImagePlaneDetails;

/**
 * @author Lee Kamentsky
 * 
 * An ImageSetError which reports that more than one ipd has the same key
 *
 */
public class ImageSetDuplicateError extends ImageSetError {
	final private List<ImagePlaneDetails> ipds;
	public ImageSetDuplicateError(
			String channelName, String message, List<String> key, 
			List<ImagePlaneDetails> ipds) {
		super(channelName, message, key);
		this.ipds = ipds;
	}
	List<ImagePlaneDetails> getImagePlaneDetails() { return ipds; }
}
