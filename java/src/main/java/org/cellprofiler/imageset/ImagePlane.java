/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */
package org.cellprofiler.imageset;

import ome.xml.model.Image;
import ome.xml.model.Pixels;
import ome.xml.model.Plane;

/**
 * @author Lee Kamentsky
 * 
 * An image plane is a 2D monochrome plane within an image file.
 * There is enough information in the ImagePlane to extract the
 * correct plane from the correct file.
 * 
 * Planes have a series # and image index and these can be used to
 * reference the particulars for the plane such as Z and T within
 * the image file metadata.
 * 
 * For interleaved images, the channels can be split out individually,
 * a monochrome image can be synthesized out of the color image
 * or the image can be left as color.
 *
 * In the context of CellProfiler, it's often not scalable to have
 * the UI read the OME XML metadata for every image during image
 * set creation. We assume that the images are what the user
 * chooses them to be and coerce them to the correct type
 * when the image is read - because of this, sometimes
 * a plane is not a plane...
 */
public class ImagePlane {
	private final ImageSeries imageSeries;
	private final int index;
	private final int channel;
	
	/**
	 * This "channel number" requests that the "image plane"
	 * should be interpreted as a labels matrix, possibly composed
	 * of overlapping objects organized as several planes.
	 * 
	 * It may be that the OME XML hasn't been collected and,
	 * at the time of reading, if several planes are discovered,
	 * they should be interpreted as overlapping objects.
	 */
	static final public int OBJECT_PLANES=-3;
	/**
	 * This "channel number" indicates that, although the
	 * data may be organized in a single plane, the plane is
	 * composed of alternating red, green and blue (and maybe alpha)
	 * values.
	 * 
	 *  Also, if a plane is marked as INTERLEAVED and there is
	 *  no OME XML for the plane, the true format should be
	 *  discovered when it is read and the image should be
	 *  coerced to a stack of color planes
	 */
	static final public int INTERLEAVED=-2;
	/**
	 * This is the "channel number" for formats where we
	 * force color images to be monochrome.
	 */
	static final public int ALWAYS_MONOCHROME=-1;
	static final public int RED_CHANNEL=0;
	static final public int GREEN_CHANNEL=1;
	static final public int BLUE_CHANNEL=2;
	static final public int ALPHA_CHANNEL=3;
	/**
	 * Construct an image plane from a file, series and index
	 * 
	 * @param imageSeries the series or OMEXML Image containing the plane
	 * @param index the index into the image stack
	 * @param channel for interleaved formats, the index of the monochrome plane
	 */
	public ImagePlane(ImageSeries imageSeries, int index, int channel) {
		this.imageSeries = imageSeries;
		this.index = index;
		this.channel = channel;
	}
	
	/**
	 * @return true if we can determine the C / Z / T position of the plane
	 */
	public boolean hasCZT() {
		return imageSeries.getOMEImage() != null; 
	}
	
	/**
	 * @return the channel index of this plane
	 */
	public int theC() {
		if (! hasCZT()) return 0;
		final Plane plane = getOMEPlane();
		if (plane != null) return plane.getTheC().getValue();
		return MetadataUtils.getC(imageSeries.getOMEImage().getPixels(), index);
	}
	
	/**
	 * @return the z-stack index of this plane
	 */
	public int theZ() {
		if (! hasCZT()) return 0;
		final Plane plane = getOMEPlane();
		if (plane != null) return plane.getTheZ().getValue();
		return MetadataUtils.getZ(imageSeries.getOMEImage().getPixels(), index);
	}
	
	/**
	 * @return the t-stack index of this plane
	 */
	public int theT() {
		if (! hasCZT()) return 0;
		final Plane plane = getOMEPlane();
		if (plane != null) return plane.getTheT().getValue();
		return MetadataUtils.getT(imageSeries.getOMEImage().getPixels(), index);
	}
	/**
	 * Construct the default monochrome image plane for a file
	 * 
	 * @param imageFile
	 */
	static public ImagePlane makeMonochromePlane(ImageFile imageFile) {
		return new ImagePlane(imageFile, ALWAYS_MONOCHROME);
	}
	
	/**
	 * Construct an interleaved color image plane for an image file.
	 * 
	 * @param imageFile
	 * @return
	 */
	static public ImagePlane makeColorPlane(ImageFile imageFile) {
		return new ImagePlane(imageFile, INTERLEAVED);
	}
	
	/**
	 * Construct an objects plane for reading labels matrices.
	 * 
	 * @param imageFile
	 * @return
	 */
	static public ImagePlane makeObjectsPlane(ImageFile imageFile) {
		return new ImagePlane(imageFile, OBJECT_PLANES);
	}
	
	/**
	 * Construct one of the color planes for an interleaved color file
	 * or create an interleaved color file using ImagePlane.INTERLEAVED
	 * 
	 * @param imageFile
	 * @param channel
	 */
	public ImagePlane(ImageFile imageFile, int channel) {
		this.imageSeries = new ImageSeries(imageFile, 0);
		this.index = 0;
		this.channel = channel;
	}
	
	/**
	 * @return the image file containing this plane
	 */
	public ImageFile getImageFile() { return imageSeries.getImageFile(); }
	
	/**
	 * @return the plane's series
	 */
	public ImageSeries getSeries() { return imageSeries; }
	
	/**
	 * @return the plane's index
	 */
	public int getIndex() { return index; }
	
	/**
	 * @return the channel index for interleaved images.
	 */
	public int getChannel() { return channel; }
	
	/**
	 * @return this plane's Plane element in the OME XML model
	 */
	public Plane getOMEPlane() {
		final Image image = imageSeries.getOMEImage();
		if (image == null) return null;
		final Pixels pixels = image.getPixels();
		if (pixels.sizeOfPlaneList() <= index) return null;
		return pixels.getPlane(index);
	}
	@Override
	public String toString() {
		return String.format("ImagePlane: %s, series=%d, index=%d, channel=%s", 
				imageSeries.getImageFile(), imageSeries.getSeries(), index,
				(channel == ALWAYS_MONOCHROME)?"Monochrome":
				((channel == INTERLEAVED)?"Color":
					((channel == OBJECT_PLANES)?"Objects":Integer.toString(channel))));
	}
}
