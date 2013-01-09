/**
 * 
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
