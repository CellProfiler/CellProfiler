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
