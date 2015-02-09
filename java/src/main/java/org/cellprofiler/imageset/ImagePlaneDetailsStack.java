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

import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import ome.xml.model.Annotation;
import ome.xml.model.Image;
import ome.xml.model.LongAnnotation;
import ome.xml.model.OME;
import ome.xml.model.Pixels;
import ome.xml.model.Plane;
import ome.xml.model.StructuredAnnotations;
import ome.xml.model.TiffData;
import ome.xml.model.UUID;
import ome.xml.model.enums.DimensionOrder;
import ome.xml.model.primitives.NonNegativeInteger;
import ome.xml.model.primitives.PositiveInteger;
import net.imglib2.meta.Axes;
import net.imglib2.meta.AxisType;
import net.imglib2.meta.TypedAxis;


/**
 * @author Lee Kamentsky
 *
 * A PlaneStack of ImagePlaneDetails. This class is mostly
 * here to give a class marker that can be used by
 * the StackAdapter filter predicate's getInputClass method.
 */
public class ImagePlaneDetailsStack extends PlaneStack<ImagePlaneDetails> {
	
	/**
	 * The "description" field for a plane's series (or image index). 
	 */
	final static public String SERIES_ANNOTATION_DESCRIPTION = "Series";
	/**
	 * The "description" field for a plane's channel
	 */
	final static public String CHANNEL_ANNOTATION_DESCRIPTION = "Channel";
	/**
	 * The annotation namepace for the series annotation.
	 */
	final static public String ANNOTATION_NAMESPACE = "info://cellprofiler.org/imageset/annotation/2014-04-17";

	final static private NonNegativeInteger NNI_ZERO = new NonNegativeInteger(0);
	
	public ImagePlaneDetailsStack(final TypedAxis... axes){
		super(axes);
	}
	/**
	 * Make a one-frame stack
	 * 
	 * @return
	 */
	static public ImagePlaneDetailsStack makeMonochromeStack(ImagePlaneDetails plane) {
		ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(XYAxes);
		stack.add(plane, 0, 0);
		return stack;
	}
	/**
	 * Make a color stack with one initial plane
	 * 
	 * @param plane
	 * @return a XYC stack containing the plane
	 */
	static public ImagePlaneDetailsStack makeColorStack(ImagePlaneDetails plane) {
		ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(XYCAxes);
		stack.add(plane, 0, 0, 0);
		return stack;
	}
	/**
	 * Make a stack of labels matrices with one initial plane
	 * 
	 * @param plane
	 * @return an XYO stack containing the plane
	 */
	static public ImagePlaneDetailsStack makeObjectsStack(ImagePlaneDetails plane) {
		ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(XYOAxes);
		stack.add(plane, 0, 0, 0);
		return stack;
		
	}
	public boolean containsKey(String key) {
		for (ImagePlaneDetails ipd:this) {
			if (ipd.containsKey(key)) return true;
		}
		return false;
	}
	public String get(String key) {
		for (ImagePlaneDetails ipd:this) {
			final String value = ipd.get(key);
			if (value != null) return value;
		}
		return null;
	}
	
	/**
	 * Add this stack to the OME root.
	 * 
	 * @param ome root of OME schema
	 * @param id the ID to use to name the stack.
	 */
	public void addToOME(OME ome, final String id) {
		final Image image = new Image();
		image.setID(id);
		final Pixels pixels = new Pixels();
		image.setPixels(pixels);
		ome.addImage(image);
		pixels.setID(String.format("Pixels:%s", id));
		final ImagePlaneDetails firstIPD = iterator().next();
		final Image srcImage = firstIPD.getImagePlane().getSeries().getOMEImage();
		PositiveInteger xSize = new PositiveInteger(1);
		PositiveInteger ySize = new PositiveInteger(1);
		if (srcImage != null) {
			final Pixels srcPixels = srcImage.getPixels();
			xSize = srcPixels.getSizeX();
			ySize = srcPixels.getSizeY();
		}
		pixels.setSizeX(xSize);
		pixels.setSizeY(ySize);
		setDimensionOrder(this, pixels);
		int nPlanes = 1;
		for (int axisIdx=0; axisIdx<numDimensions(); axisIdx++) {
			TypedAxis a = axis(axisIdx);
			final PositiveInteger size = new PositiveInteger(size(axisIdx));
			if (a.type().equals(Axes.Z)) {
				pixels.setSizeZ(size);
			} else if (a.type().equals(Axes.CHANNEL)) {
				pixels.setSizeC(size);
			} else {
				// Both T and ObjectPlane go here
				pixels.setSizeT(size);
			}
			nPlanes = nPlanes * size(axisIdx);
		}
		int [] coords = new int [numDimensions()];
		int c = 0;
		int z = 0;
		int t = 0;
		for (int planeIdx=0; planeIdx < nPlanes; planeIdx++ ) {
			final ImagePlaneDetails ipd = get(coords);
			addOMEPlane(ome, pixels, xSize, ySize, c, z, t, ipd);
			
			// Advance to the next coordinate.
			boolean done = false;
			for (int j=2; (j< coords.length) && ! done; j++) {
				if (coords[j] == size(j)-1) {
					coords[j] = 0;
				} else {
					coords[j]++;
					done = true;
				}
				final AxisType axisType = axis(j).type();
				if (axisType.equals(Axes.CHANNEL) ) {
					c = coords[j];
				} else if (axisType.equals(Axes.Z)) {
					z = coords[j];
				} else {
					t = coords[j];
				}
			}
		}
	}
	
	/**
	 * Load the stack from an OME Image
	 * 
	 * @param ome the OME root node
	 * @param id the ID of the Image node to load from  
	 * @throws URISyntaxException 
	 */
	public void loadFromOME(OME ome, String id) throws URISyntaxException {
		for (int i=0; i<ome.sizeOfImageList(); i++) {
			Image omeImage = ome.getImage(i);
			if (omeImage.getID().equals(id)) {
				loadFromOMEImage(omeImage);
				return;
			}
		}
		throw new IndexOutOfBoundsException(String.format("Could not find image, \"%s\" in OME-XML", id));
	}
	
	/**
	 * Load a stack from an OME Image node
	 * 
	 * @param omeImage
	 * @throws URISyntaxException
	 */
	private void loadFromOMEImage(Image omeImage) throws URISyntaxException {
		Map<URI, ImageFileDetails> imageFiles = new HashMap<URI, ImageFileDetails>();
		Map<URI, Map<Integer, ImageSeriesDetails>> imageSeries = new HashMap<URI, Map<Integer,ImageSeriesDetails>>();
		final Pixels pixels = omeImage.getPixels();
		int [] coords = new int[numDimensions()];
		for (int planeIdx=0; planeIdx<pixels.sizeOfPlaneList() && planeIdx<pixels.sizeOfTiffDataList(); planeIdx++) {
			final Plane plane = pixels.getPlane(planeIdx);
			final TiffData location = pixels.getTiffData(planeIdx);
			final URI uri = new URI(location.getUUID().getFileName());
			if (! imageFiles.containsKey(uri)) {
				imageFiles.put(uri, new ImageFileDetails(new ImageFile(uri)));
			}
			final ImageFileDetails ifd = imageFiles.get(uri);
			final int series = getLongAnnotationFromPlane(plane, SERIES_ANNOTATION_DESCRIPTION, 0);
			if (! imageSeries.containsKey(uri)) {
				imageSeries.put(uri, new HashMap<Integer, ImageSeriesDetails>());
			}
			Map<Integer, ImageSeriesDetails> idx2Series = imageSeries.get(uri);
			if (! idx2Series.containsKey(series)) {
				idx2Series.put(series, new ImageSeriesDetails(new ImageSeries(ifd.getImageFile(), series), ifd));
			}
			final ImageSeriesDetails isd = idx2Series.get(series);
			final int idx = location.getIFD().getValue();
			final int channel = getLongAnnotationFromPlane(plane, CHANNEL_ANNOTATION_DESCRIPTION, ImagePlane.ALWAYS_MONOCHROME);
			final ImagePlane imagePlane = new ImagePlane(isd.getImageSeries(), idx, channel);
			final ImagePlaneDetails ipd = new ImagePlaneDetails(imagePlane, isd);
			for (int didx=0; didx<numDimensions();didx++) {
				AxisType at = axis(didx).type();
				if (at.equals(Axes.CHANNEL)) {
					coords[didx] = plane.getTheC().getValue();
				} else if (at.equals(Axes.Z)) {
					coords[didx] = plane.getTheZ().getValue();
				} else if (at.equals(Axes.TIME)) {
					coords[didx] = plane.getTheT().getValue();
				} else if (at.equals(OBJECT_PLANE_AXIS_TYPE)) {
					coords[didx] = planeIdx;
				}
			}
			add(ipd, coords);
		}
	}
	
	/**
	 * Add an image plane to the "pixels" of an OME image
	 * 
	 * @param ome the root node of the OME document
	 * @param pixels add the plane to these pixels
	 * @param xSize the size of a plane in the X direction (not definitive)
	 * @param ySize the size of a plane in the Y direction
	 * @param c the channel index of the plane
	 * @param z the z index of the plane
	 * @param t the time index of the plane
	 * @param ipd the image plane that's the source of the pixels.
	 */
	private static void addOMEPlane(OME ome, final Pixels pixels,
			PositiveInteger xSize, PositiveInteger ySize, int c, int z, int t,
			final ImagePlaneDetails ipd) {
		final Plane destPlane = new Plane();
		pixels.addPlane(destPlane);
		int series = 0;
		int channel = ImagePlane.ALWAYS_MONOCHROME;
		final TiffData location = new TiffData();
		final UUID uuid = new UUID();
		location.setPlaneCount(new NonNegativeInteger(1));
		if (ipd != null) {
			final ImagePlane imagePlane = ipd.getImagePlane();
			series = imagePlane.getSeries().getSeries();
			channel = imagePlane.getChannel();
			location.setIFD(new NonNegativeInteger(imagePlane.getIndex()));
			uuid.setFileName(imagePlane.getImageFile().getURI().toString());
			final Plane omePlane = imagePlane.getOMEPlane();
			if (omePlane != null) {
				location.setFirstC(omePlane.getTheC());
				location.setFirstT(omePlane.getTheT());
				location.setFirstZ(omePlane.getTheZ());
			} else {
				location.setFirstC(NNI_ZERO);
				location.setFirstT(NNI_ZERO);
				location.setFirstZ(NNI_ZERO);
				
			}
		} else {
			// A missing plane.
			// This retrieves a URI placeholder for it.
			uuid.setFileName(new ImageFile(xSize.getValue(), ySize.getValue()).getURI().toString());
			location.setIFD(NNI_ZERO);
			location.setFirstC(NNI_ZERO);
			location.setFirstT(NNI_ZERO);
			location.setFirstZ(NNI_ZERO);
		}
		location.setUUID(uuid);
		pixels.addTiffData(location);
		destPlane.setTheC(new NonNegativeInteger(c));
		destPlane.setTheZ(new NonNegativeInteger(z));
		destPlane.setTheT(new NonNegativeInteger(t));
		if (series > 0) {
			addLongAnnotationToPlane(ome, destPlane, SERIES_ANNOTATION_DESCRIPTION, series);
		}
		if (channel != ImagePlane.ALWAYS_MONOCHROME)
			addLongAnnotationToPlane(ome, destPlane, CHANNEL_ANNOTATION_DESCRIPTION, channel);
	}
	/**
	 * Read a numeric annotation associated with the given plane
	 * 
	 * @param plane the possibly annotated plane
	 * @param description the annotation description
	 * @param defaultValue the value to return if the annotation isn't present.
	 * @return
	 */
	private static int getLongAnnotationFromPlane(Plane plane, String description, int defaultValue) {
		int result = defaultValue;
		for (int annotationRefIdx=0; annotationRefIdx<plane.sizeOfLinkedAnnotationList(); annotationRefIdx++) {
			Annotation a = plane.getLinkedAnnotation(annotationRefIdx);
			if (a.getDescription().equals(description) &&
				a.getNamespace().equals(ANNOTATION_NAMESPACE) &&
				a instanceof LongAnnotation) {
				result = ((LongAnnotation)a).getValue().intValue();
				break;
			}
		}
		return result;
	}
	/**
	 * Add an integer annotation to an image plane, such as series or channel
	 * 
	 * @param ome the root node of the document
	 * @param plane the plane to be annotated
	 * @param description the annotation name
	 * @param value the value to assign
	 */
	private static void addLongAnnotationToPlane(OME ome, Plane plane, String description, long value) {
		// Non-standard, have to add a "Series" annotation
		//
		// First, see if we have one for this series before
		// creating a new one
		//
		StructuredAnnotations sa = ome.getStructuredAnnotations();
		if (sa == null) {
			ome.setStructuredAnnotations(sa = new StructuredAnnotations());
		}
		LongAnnotation seriesAnnotation = null;
		for (int saIdx=0; saIdx<sa.sizeOfLongAnnotationList(); saIdx++) {
			final LongAnnotation candidate = sa.getLongAnnotation(saIdx);
			if ((candidate.getDescription().equals(description)) &&
				(candidate.getNamespace().equals(ANNOTATION_NAMESPACE)) &&
				(candidate.getValue().equals(value))) {
				seriesAnnotation = candidate;
				break;
			}
		}
		if (seriesAnnotation == null) {
			seriesAnnotation = new LongAnnotation();
			seriesAnnotation.setDescription(description);
			seriesAnnotation.setNamespace(ANNOTATION_NAMESPACE);
			seriesAnnotation.setValue(value);
			seriesAnnotation.setID(String.format("%s/%s/%d", ANNOTATION_NAMESPACE, description, value));
			sa.addLongAnnotation(seriesAnnotation);
		}
		plane.linkAnnotation(seriesAnnotation);
		
	}
	/**
	 * @param ipds
	 * @param pixels
	 */
	private static void setDimensionOrder(final ImagePlaneDetailsStack ipds,
			final Pixels pixels) {
		//
		// Determine the dimension order
		//
		if (ipds.numDimensions() == 2) {
			pixels.setDimensionOrder(DimensionOrder.XYCZT);
		} else {
			final AxisType type = ipds.axis(2).type();
			if (type.equals(Axes.CHANNEL)) {
				if (ipds.numDimensions() == 3) {
					pixels.setDimensionOrder(DimensionOrder.XYCZT);
				} else if (ipds.axis(3).type().equals(Axes.Z)) {
					pixels.setDimensionOrder(DimensionOrder.XYCZT);
				} else {
					pixels.setDimensionOrder(DimensionOrder.XYCTZ);
				}
			} else if (type.equals(Axes.Z)) {
				if (ipds.numDimensions() == 3) {
					pixels.setDimensionOrder(DimensionOrder.XYZCT);
				} else if (ipds.axis(3).type().equals(Axes.CHANNEL)) {
					pixels.setDimensionOrder(DimensionOrder.XYZCT);
				} else {
					pixels.setDimensionOrder(DimensionOrder.XYZTC);
				}
			} else if ((ipds.numDimensions() == 3) || (ipds.axis(3).type().equals(Axes.CHANNEL))) {
				pixels.setDimensionOrder(DimensionOrder.XYTCZ);
			} else {
				pixels.setDimensionOrder(DimensionOrder.XYTZC);
			}
		}
	}
	
}
