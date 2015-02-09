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

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.util.HashSet;
import java.util.Set;

import javax.xml.parsers.ParserConfigurationException;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;

import org.cellprofiler.imageset.filter.Filter;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;
import org.xml.sax.SAXException;

/**
 * @author Lee Kamentsky
 *
 */
public class TestImagePlaneMetadataExtractor {
	private ImagePlane makeImagePlane(String path, String filename, String omexml, int series, int index, int channel) {
		File file = new File(new File(System.getProperty("user.home"), path), filename);
		try {
			ImageFile imageFile = new ImageFile(file.toURI());
			if (omexml != null) imageFile.setXMLDocument(omexml);
			return new ImagePlane(new ImageSeries(imageFile, series), index, channel);
		} catch (MalformedURLException e) {
			fail();
		} catch (ParserConfigurationException e) {
			fail();
		} catch (SAXException e) {
			fail();
		} catch (IOException e) {
			fail();
		} catch (DependencyException e) {
			fail();
		} catch (ServiceException e) {
			fail();
		}
		return null;
	}
	
	private ImagePlane makeImagePlane(String path, String filename) {
		return makeImagePlane(path, filename, null, 0, 0, 0);
	}

	@Test
	public void testNothing() {
		ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
		assertFalse(x.extract(makeImagePlane("foo", "bar.jpg")).iterator().hasNext());
	}
	
	@Test
	public void testFileRegexp() {
		ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
		x.addFileNameRegexp("(?P<WellName>[A-H][0-9]{2})");
		ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
		assertTrue(ipd.containsKey("WellName"));
		assertEquals(ipd.get("WellName"), "A01");
	}
	
	@Test
	public void testFilteredFileRegexp() {
		try {
			Filter<ImageFile> filter = 
				new Filter<ImageFile>("directory does contain \"Plate1\"", ImageFile.class);
			ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
			x.addFileNameRegexp("(?P<WellName>[A-H][0-9]{2})", filter);
			ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
			assertTrue(ipd.containsKey("WellName"));
			assertEquals(ipd.get("WellName"), "A01");
			ipd = x.extract(makeImagePlane("Plate2", "A01.tif"));
			assertFalse(ipd.containsKey("WellName"));
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
	@Test
	public void testPathRegexp() {
		ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
		x.addPathNameRegexp("(?P<Plate>Plate[0-9]+)");
		ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
		assertTrue(ipd.containsKey("Plate"));
		assertEquals(ipd.get("Plate"), "Plate1");
	}
	
	@Test
	public void testFilteredPathRegexp() {
		try {
			Filter<ImageFile> filter = 
				new Filter<ImageFile>("file does contain \"A01\"", ImageFile.class);
			ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
			x.addPathNameRegexp("(?P<Plate>Plate[0-9]+)", filter);
			ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
			assertTrue(ipd.containsKey("Plate"));
			assertEquals(ipd.get("Plate"), "Plate1");
			ipd = x.extract(makeImagePlane("Plate1", "A02.tif"));
			assertFalse(ipd.containsKey("Plate"));
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
	
	@Test
	public void testExtractMetadataOME() {
		/*
		 * Test the big extract method
		 * 
		 * We use OME metadata for a 3-series, 36 z-stack file
		 * and we conditionally extract file name metadata.
		 */
		String xml = TestOMEMetadataExtractor.getTestXMLAsString();
		try {
			ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
			Filter<ImageFile> filter = new Filter<ImageFile>(
					"file does contain \"A02\"", ImageFile.class);
			x.addImagePlaneExtractor(new OMEPlaneMetadataExtractor());
			x.addImageSeriesExtractor(new OMESeriesMetadataExtractor());
			x.addFileNameRegexp("(?P<WellName>[A-H][0-9]{2})", filter);
			String [] urls = {
					new File(System.getProperty("user.home"), "Image_A01.tif").toURI().toString(),
					new File(System.getProperty("user.home"), "Image_A02.tif").toURI().toString() };
			String [] omeMetadata = { xml, xml };
			Set<String> keysOut = new HashSet<String>();
					
			ImagePlaneDetails [] ipds = x.extract(urls, omeMetadata, keysOut);
			for (String key:new String [] {"T", "Z", "ChannelName", "WellName"}) {
				assertTrue(keysOut.contains(key));
			}
			assertEquals(ipds.length, 36 * 4 * 2);
			boolean [][][] found = {
					{ new boolean[36], new boolean[36], new boolean[36], new boolean[36] },
					{ new boolean[36], new boolean[36], new boolean[36], new boolean[36] }
			};
			for (ImagePlaneDetails ipd:ipds) {
				int imageIndex = ipd.getImagePlane().getImageFile().getFileName().equals("Image_A01.tif")?0:1;
				int seriesIndex = ipd.getImagePlane().getSeries().getSeries();
				int zIndex = ipd.getImagePlane().getOMEPlane().getTheZ().getValue();
				assertEquals("0", ipd.get("T"));
				assertEquals(zIndex, Integer.valueOf(ipd.get("Z")).intValue());
				assertEquals("Exp1Cam1", ipd.get("ChannelName"));
				assertEquals(imageIndex == 1, ipd.containsKey("WellName"));
				assertEquals("136570140804 96_Greiner", ipd.get("Plate"));
				assertEquals("E11", ipd.get("Well"));
				found[imageIndex][seriesIndex][zIndex] = true;
			}
			for (boolean [][] fff:found) {
				for(boolean [] ff:fff) {
					for (boolean f:ff) assertTrue(f);
				}
			}
			
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (ParserConfigurationException e) {
			fail();
		} catch (SAXException e) {
			fail();
		} catch (IOException e) {
			fail();
		} catch (DependencyException e) {
			fail();
		} catch (ServiceException e) {
			fail();
		} catch (URISyntaxException e) {
			fail();
		}
	}
	@Test
	public void testExtractMetadataNoOME() {
		try {
			ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
			Filter<ImageFile> filter = new Filter<ImageFile>(
					"file does contain \"A02\"", ImageFile.class);
			x.addImagePlaneExtractor(new OMEPlaneMetadataExtractor());
			x.addImageSeriesExtractor(new OMESeriesMetadataExtractor());
			x.addFileNameRegexp("(?P<WellName>[A-H][0-9]{2})", filter);
			String [] urls = {
					new File(System.getProperty("user.home"), "Image_A01.tif").toURI().toString(),
					new File(System.getProperty("user.home"), "Image_A02.tif").toURI().toString() };
			Set<String> keysOut = new HashSet<String>();
					
			ImagePlaneDetails [] ipds = x.extract(urls, new String [2], keysOut);
			for (String key:new String [] {"WellName"}) {
				assertTrue(keysOut.contains(key));
			}
			assertEquals(ipds.length,  urls.length);
			assertFalse(ipds[0].containsKey("WellName"));
			assertEquals(ipds[1].get("WellName"), "A02");
			for (int i=0; i<urls.length; i++) {
				ImagePlaneDetails ipd = ipds[i];
				ImagePlane plane = ipd.getImagePlane();
				ImageSeries series = plane.getSeries();
				ImageFile file = series.getImageFile();
				assertEquals(file.getURI().toString(), urls[i]);
				assertEquals(plane.getIndex(), 0);
				assertEquals(plane.getChannel(), ImagePlane.ALWAYS_MONOCHROME);
				assertEquals(series.getSeries(), 0);
			}
		} catch (BadFilterExpressionException e) {
			e.printStackTrace();
			fail();
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
			fail();
		} catch (SAXException e) {
			e.printStackTrace();
			fail();
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		} catch (DependencyException e) {
			e.printStackTrace();
			fail();
		} catch (ServiceException e) {
			e.printStackTrace();
			fail();
		}		
	}
}
