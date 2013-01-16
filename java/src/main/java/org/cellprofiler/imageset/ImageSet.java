/**
 * 
 */
package org.cellprofiler.imageset;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.cellprofiler.imageset.filter.ImagePlaneDetails;

/**
 * @author Lee Kamentsky
 *
 * An ImageSet is a collection of ImagePlanes coallated for
 * processing during a CellProfiler cycle.
 */
public class ImageSet extends ArrayList<ImagePlaneDetails> {
	private static final long serialVersionUID = -6824821112413090930L;
	final private List<String> key;
	/**
	 * Construct the image set from its image plane descriptors and key
	 * @param ipds
	 * @param key
	 */
	public ImageSet(Collection<ImagePlaneDetails> ipds, List<String> key) {
		super(ipds);
		this.key = key;
	}
	
	public List<String> getKey() {
		return key;
	}
	
}
