/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2013 Broad Institute
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

import javax.xml.parsers.ParserConfigurationException;

import org.cellprofiler.imageset.filter.Filter;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.cellprofiler.imageset.filter.ImagePlaneDetails;
import org.junit.Test;
import org.xml.sax.SAXException;

/**
 * @author Lee Kamentsky
 *
 */
public class TestImagePlaneMetadataExtractor {
	private ImagePlane makeImagePlane(String path, String filename, String omexml, int series, int index) {
		File file = new File(new File(System.getProperty("user.home"), path), filename);
		try {
			ImageFile imageFile = new ImageFile(file.toURI().toURL());
			if (omexml != null) imageFile.setXMLDocument(omexml);
			return new ImagePlane(imageFile, series, index);
		} catch (MalformedURLException e) {
			fail();
		} catch (ParserConfigurationException e) {
			fail();
		} catch (SAXException e) {
			fail();
		} catch (IOException e) {
			fail();
		}
		return null;
	}
	
	private ImagePlane makeImagePlane(String path, String filename) {
		return makeImagePlane(path, filename, null, 0, 0);
	}

	@Test
	public void testNothing() {
		ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
		assertEquals(0, x.extract(makeImagePlane("foo", "bar.jpg")).metadata.size());
	}
	
	@Test
	public void testFileRegexp() {
		ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
		x.addFileNameRegexp("(?P<WellName>[A-H][0-9]{2})");
		ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
		assertTrue(ipd.metadata.containsKey("WellName"));
		assertEquals(ipd.metadata.get("WellName"), "A01");
	}
	
	@Test
	public void testFilteredFileRegexp() {
		try {
			Filter filter = new Filter("directory does contain \"Plate1\"");
			ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
			x.addFileNameRegexp("(?P<WellName>[A-H][0-9]{2})", filter);
			ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
			assertTrue(ipd.metadata.containsKey("WellName"));
			assertEquals(ipd.metadata.get("WellName"), "A01");
			ipd = x.extract(makeImagePlane("Plate2", "A01.tif"));
			assertFalse(ipd.metadata.containsKey("WellName"));
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
	@Test
	public void testPathRegexp() {
		ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
		x.addPathNameRegexp("(?P<Plate>Plate[0-9]+)");
		ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
		assertTrue(ipd.metadata.containsKey("Plate"));
		assertEquals(ipd.metadata.get("Plate"), "Plate1");
	}
	
	@Test
	public void testFilteredPathRegexp() {
		try {
			Filter filter = new Filter("file does contain \"A01\"");
			ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
			x.addPathNameRegexp("(?P<Plate>Plate[0-9]+)", filter);
			ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
			assertTrue(ipd.metadata.containsKey("Plate"));
			assertEquals(ipd.metadata.get("Plate"), "Plate1");
			ipd = x.extract(makeImagePlane("Plate1", "A02.tif"));
			assertFalse(ipd.metadata.containsKey("Plate"));
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
	
	@Test
	public void testTwoExtractors() {
		/*
		 * Make sure that the metadata from one operation is available
		 * to the filter of the second one.
		 */
		try {
			Filter filter = new Filter("metadata does WellName \"A01\"");
			ImagePlaneMetadataExtractor x = new ImagePlaneMetadataExtractor();
			x.addFileNameRegexp("(?P<WellName>[A-H][0-9]{2})");
			x.addPathNameRegexp("(?P<Plate>Plate[0-9]+)", filter);
			ImagePlaneDetails ipd = x.extract(makeImagePlane("Plate1", "A01.tif"));
			assertTrue(ipd.metadata.containsKey("Plate"));
			assertEquals(ipd.metadata.get("Plate"), "Plate1");
			ipd = x.extract(makeImagePlane("Plate1", "A02.tif"));
			assertFalse(ipd.metadata.containsKey("Plate"));
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
}
