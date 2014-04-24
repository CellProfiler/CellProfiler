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

import java.io.File;
import java.net.URI;
import java.util.List;

import ome.xml.model.LongAnnotation;
import ome.xml.model.OME;
import ome.xml.model.Pixels;
import ome.xml.model.Plane;
import ome.xml.model.StructuredAnnotations;
import ome.xml.model.TiffData;
import ome.xml.model.UUID;
import ome.xml.model.primitives.NonNegativeInteger;
import ome.xml.model.primitives.PositiveInteger;


/**
 * @author Lee Kamentsky
 *
 * The ImagePlaneDetails class couples an ImagePlane to the
 * metadata extracted from the ImagePlane by the extractors.
 */
public class ImagePlaneDetails extends Details {
	/**
	 * The image plane to be filtered 
	 */
	private final ImagePlane imagePlane;
	public ImagePlaneDetails(ImagePlane imagePlane, Details parent) {
		super(parent);
		this.imagePlane = imagePlane;
	}
	protected ImagePlaneDetails(ImagePlane imagePlane, Details parent, Details child) {
		super(parent, child);
		this.imagePlane = imagePlane;
	}
	public ImagePlane getImagePlane() {
		return imagePlane;
	}
	
	/**
	 * Sometimes it is necessary to force a color interpretation
	 * on an image plane details by cloning it and replacing
	 * the image plane with one of a different interpretation
	 * 
	 * (insert emoji of cat throwing up please)
	 * 
	 * @return either self if already monochrome or a clone
	 */
	public ImagePlaneDetails coerceToMonochrome() {
		if (imagePlane.getChannel() == ImagePlane.INTERLEAVED) {
			final ImagePlane planeCopy = new ImagePlane(
					imagePlane.getSeries(), imagePlane.getIndex(), ImagePlane.ALWAYS_MONOCHROME);
			return new ImagePlaneDetails(planeCopy, parent, this);
		} else {
			return this;
		}
	}
	/**
	 * Sometimes it is necessary to force a color interpretation
	 * on an image plane details by cloning it and replacing
	 * the image plane with one of a different interpretation
	 * 
	 * (insert emoji of cat throwing up please)
	 * 
	 * @return either self if already monochrome or a clone
	 */
	public ImagePlaneDetails coerceToColor() {
		if (imagePlane.getChannel() != ImagePlane.INTERLEAVED) {
			final ImagePlane planeCopy = new ImagePlane(
					imagePlane.getSeries(), imagePlane.getIndex(), ImagePlane.INTERLEAVED);
			return new ImagePlaneDetails(planeCopy, parent, this);
		} else {
			return this;
		}
	}
	/**
	 * Get the following fields for display by CellProfiler's
	 * metadata module:
	 * Path / URL
	 * Series
	 * Index
	 * metadata as given by the metadataKeys
	 * 
	 * @param metadataKeys
	 * @return
	 */
	public String [] getIPDFields(List<String> metadataKeys) {
		final String [] result = new String[metadataKeys.size() + 3];
		URI uri = imagePlane.getImageFile().getURI();
		if (uri.getScheme().toLowerCase().equals("file")) {
			result[0] = new File(uri).toString();
		} else {
			result[0] = uri.toString();
		}
		result[1] = Integer.toString(imagePlane.getSeries().getSeries());
		result[2] = Integer.toString(imagePlane.getIndex());
		for (int i=0; i<metadataKeys.size(); i++) {
			result[i+3] = get(metadataKeys.get(i));
			if (result[i+3] == null) result[i+3] = "";
		}
		return result;
	}
}
