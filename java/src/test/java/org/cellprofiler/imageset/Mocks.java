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
import java.util.ArrayList;
import java.util.List;

import ome.xml.model.Channel;
import ome.xml.model.Image;
import ome.xml.model.OME;
import ome.xml.model.Pixels;
import ome.xml.model.Plane;
import ome.xml.model.Plate;
import ome.xml.model.Well;
import ome.xml.model.WellSample;
import ome.xml.model.enums.DimensionOrder;
import ome.xml.model.primitives.NonNegativeInteger;
import ome.xml.model.primitives.PositiveInteger;

/**
 * @author Lee Kamentsky
 *
 * Mock objets for testing
 */
public class Mocks {
	
	public static ImagePlaneDetails makeMockIPD(String filename) {
		final ImageFileDetails imageFileDetails = makeMockImageFileDetails(filename);
		final ImagePlane plane = ImagePlane.makeMonochromePlane(imageFileDetails.getImageFile());
		return new ImagePlaneDetails(plane, new ImageSeriesDetails(plane.getSeries(), imageFileDetails));
	}
	
	public static class MockImageDescription {
		final String plate;
		final String well;
		final int site;
		final int sizeX;
		final int sizeY;
		final int sizeC;
		final int sizeZ;
		final int sizeT;
		final DimensionOrder order;
		final String [] channelNames;
		static final int DEFAULT_SIZE_X = 640;
		static final int DEFAULT_SIZE_Y = 480;
		static final String [] DEFAULT_CHANNEL_NAMES = { "DNA", "GFP", "Actin" };
		static final DimensionOrder DEFAULT_DIMENSION_ORDER = DimensionOrder.XYCTZ;
		public MockImageDescription(String plate, String well, int site, int sizeX, int sizeY, int sizeC, int sizeZ, int sizeT, DimensionOrder order, String... channelNames) {
			this.plate = plate;
			this.well = well;
			this.site = site;
			this.sizeX = sizeX;
			this.sizeY = sizeY;
			this.sizeC = sizeC;
			this.sizeZ = sizeZ;
			this.sizeT = sizeT;
			this.order = order;
			this.channelNames = channelNames;
		}
		public MockImageDescription(String plate, String well, int site, int sizeX, int sizeY, int sizeC, int sizeZ, int sizeT, String... channelNames) {
			this(plate, well, site, sizeX, sizeY, sizeC, sizeZ, sizeT, DEFAULT_DIMENSION_ORDER, channelNames);
		}
		public static MockImageDescription makeColorDescription(String plate, String well, int site) {
			return new MockImageDescription(plate, well, site, DEFAULT_SIZE_X, DEFAULT_SIZE_Y, 3, 1, 1, DEFAULT_CHANNEL_NAMES);
		}
	}
	/**
	 * Make a mock ImageFile with a mock OMEXML structure including
	 * possible multiple images and plate/site/well.
	 * @param filename
	 * @param structure
	 * @return
	 */
	public static ImageFile makeMockFile(String filename, MockImageDescription...descriptions) {
		final File f = new File(new File(System.getProperty("user.home")), filename);
		final ImageFile imageFile = new ImageFile(f.toURI());
		if (descriptions.length == 0) return imageFile;
		
		OME ome = new OME();
		List<Plate> plates = new ArrayList<Plate>();
		for (MockImageDescription d:descriptions) {
			Plate plate = null;
			for (Plate qPlate:plates) {
				if (qPlate.getName().equals(d.plate)) {
					plate = qPlate;
					break;
				}
			}
			if (plate == null) {
				plate = new Plate();
				plate.setName(d.plate);
				ome.addPlate(plate);
			}
			Well well = null;
			int row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".indexOf(d.well.substring(0, 1).toUpperCase());
			int col = Integer.valueOf(d.well.substring(1));
			for (int i=0; i<plate.sizeOfWellList(); i++) {
				final Well qWell = plate.getWell(i);
				if ((qWell.getRow().equals(row)) && (qWell.getColumn().equals(col))) {
					well = qWell;
					break;
				}
			}
			if (well == null) {
				well = new Well();
				well.setRow(new NonNegativeInteger(row));
				well.setColumn(new NonNegativeInteger(col));
				well.setExternalDescription(d.well);
				plate.addWell(well);
			}
			while (well.sizeOfWellSampleList() <= d.site) {
				final WellSample qSample = new WellSample();
				qSample.setIndex(new NonNegativeInteger(well.sizeOfWellSampleList()));
				well.addWellSample(qSample);
			}
			final WellSample sample = well.getWellSample(d.site);
			Image image = new Image();
			Pixels pixels = new Pixels();
			Channel [] channels = new Channel[d.sizeC];
			for (int i=0; i<d.sizeC; i++) {
				Channel c = new Channel();
				c.setName(MockImageDescription.DEFAULT_CHANNEL_NAMES[i]);
				pixels.addChannel(c);
				channels[i] = c;
			}
			pixels.setSizeX(new PositiveInteger(d.sizeX));
			pixels.setSizeY(new PositiveInteger(d.sizeY));
			pixels.setSizeC(new PositiveInteger(d.sizeC));
			pixels.setSizeZ(new PositiveInteger(d.sizeZ));
			pixels.setSizeT(new PositiveInteger(d.sizeT));
			pixels.setDimensionOrder(d.order);
			for (int i=0; i<d.sizeC*d.sizeZ*d.sizeT; i++) {
				int c=0, z=0, t=0;
				switch(d.order) {
				case XYCTZ:
					c = i % d.sizeC;
					t = (i / d.sizeC) % d.sizeT;
					z = i / d.sizeC / d.sizeT;
					break;
				case XYCZT:
					c = i % d.sizeC;
					z = (i / d.sizeC) % d.sizeZ;
					t = i / d.sizeC / d.sizeZ;
					break;
				case XYTCZ:
					t = i % d.sizeT;
					c = (i / d.sizeT) % d.sizeC;
					z = i / d.sizeC / d.sizeT;
					break;
				case XYTZC:
					t = i % d.sizeT;
					z = (i / d.sizeT) % d.sizeZ;
					c = i / d.sizeZ / d.sizeT;
					break;
				case XYZCT:
					z = i % d.sizeZ;
					c = (i / d.sizeZ) % d.sizeC;
					t = i / d.sizeC / d.sizeZ;
					break;
				case XYZTC:
					z = i % d.sizeZ;
					t = (i / d.sizeZ) % d.sizeT;
					c = i / d.sizeT / d.sizeZ;
					break;
				}
				Plane plane = new Plane();
				plane.setTheC(new NonNegativeInteger(c));
				plane.setTheZ(new NonNegativeInteger(z));
				plane.setTheT(new NonNegativeInteger(t));
				pixels.addPlane(plane);
			}
			image.setPixels(pixels);
			sample.linkImage(image);
			ome.addImage(image);
		}
		imageFile.setXMLDocument(ome);
		return imageFile;
	}

	/**
	 * @param filename
	 * @return
	 */
	public static ImageFileDetails makeMockImageFileDetails(String filename, MockImageDescription...descriptions) {
		final ImageFile imageFile = makeMockFile(filename, descriptions);
		final ImageFileDetails imageFileDetails = new ImageFileDetails(imageFile);
		return imageFileDetails;
	}
	public static ImageFileDetails makeMockImageFileDetails() {
		return makeMockImageFileDetails("foo.tif");
	}
	
	public static ImagePlaneDetails makeMockIPD() {
		return makeMockIPD("foo.tif");
	}
	public static ImagePlaneDetailsStack makeMockColorStack(int nPlanes) {
		return makeMockColorStack("foo.tif", nPlanes, "P-12345", "A01", 0);
	}
	
	public static ImagePlaneDetailsStack makeMockColorStack(String filename, int nPlanes, String plate, String well, int site) {
		final ImageFileDetails imageFileDetails = makeMockImageFileDetails(
				filename, MockImageDescription.makeColorDescription(plate, well, site));
		return makeMockColorStack(imageFileDetails, nPlanes);
	}
	
	public static ImagePlaneDetailsStack makeMockColorStack(ImageFileDetails imageFileDetails, int nPlanes) {
		final ImageSeries imageSeries = new ImageSeries(imageFileDetails.getImageFile(), 0);
		final ImageSeriesDetails imageSeriesDetails = new ImageSeriesDetails(imageSeries, imageFileDetails);
		final ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
		for (int i=0; i<nPlanes; i++) {
			final ImagePlane imagePlane = new ImagePlane(imageSeries, i, ImagePlane.ALWAYS_MONOCHROME);
			stack.add(new ImagePlaneDetails(imagePlane, imageSeriesDetails), 0, 0, i);
		}
		return stack;
	}

	public static ImagePlaneDetailsStack makeMockInterleavedStack() {
		final ImageFileDetails imageFileDetails = makeMockImageFileDetails();
		final ImageSeries imageSeries = new ImageSeries(imageFileDetails.getImageFile(), 0);
		final ImageSeriesDetails imageSeriesDetails = new ImageSeriesDetails(imageSeries, imageFileDetails);
		final ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
		final ImagePlane imagePlane = new ImagePlane(imageSeries, 0, ImagePlane.INTERLEAVED);
		stack.add(new ImagePlaneDetails(imagePlane, imageSeriesDetails), 0, 0, 0);
		return stack;
	}
	public static ImagePlaneDetailsStack makeMockMonochromeStack() {
		final ImageFileDetails imageFileDetails = makeMockImageFileDetails();
		final ImageSeries imageSeries = new ImageSeries(imageFileDetails.getImageFile(), 0);
		final ImageSeriesDetails imageSeriesDetails = new ImageSeriesDetails(imageSeries, imageFileDetails);
		final ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYAxes);
		final ImagePlane imagePlane = new ImagePlane(imageSeries, 0, ImagePlane.ALWAYS_MONOCHROME);
		stack.add(new ImagePlaneDetails(imagePlane, imageSeriesDetails), 0, 0);
		return stack;
	}
	/**
	 * Generate a list of the IPDs for a mock file
	 * 
	 * @param filename the name of the mock file
	 * @param descriptions descriptions of each OME Image within the file
	 * 
	 * @return a list of IPDs that reflect the contents of the series
	 *         within the file, as described by the descriptions.
	 */
	public static List<ImagePlaneDetails> makeMockIPDs(String filename, MockImageDescription ... descriptions) {
		final ImageFileDetails imageFileDetails = makeMockImageFileDetails(filename, descriptions);
		final List<ImagePlaneDetails> result = new ArrayList<ImagePlaneDetails>();
		final ImageFile imageFile = imageFileDetails.getImageFile();
		final OME imageFileMetadata = imageFile.getMetadata();
		for (int series=0; series < imageFileMetadata.sizeOfImageList(); series++) {
			final ImageSeries imageSeries = new ImageSeries(imageFile, series);
			final ImageSeriesDetails imageSeriesDetails = new ImageSeriesDetails(imageSeries, imageFileDetails);
			final Pixels pixels = imageSeries.getOMEImage().getPixels();
			for (int index=0; index<pixels.sizeOfPlaneList(); index++) {
				final ImagePlane imagePlane = new ImagePlane(imageSeries, index, ImagePlane.ALWAYS_MONOCHROME);
				result.add(new ImagePlaneDetails(imagePlane, imageSeriesDetails));
			}
		}
		return result;
	}
}
