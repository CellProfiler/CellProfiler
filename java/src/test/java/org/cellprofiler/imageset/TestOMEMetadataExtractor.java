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
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.CharBuffer;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;

import org.junit.Test;
import org.xml.sax.SAXException;

/**
 * @author Lee Kamentsky
 *
 */
public class TestOMEMetadataExtractor {

	/**
	 * Test method for {@link org.cellprofiler.imageset.OMEPlaneMetadataExtractor#extract(org.cellprofiler.imageset.ImagePlane)}.
	 */
	@Test
	public void testExtract() {
		final String filename = "foo.jpg";
		File path = new File(new File(System.getProperty("user.home")), filename);
		try {
			ImageFile imageFile = new ImageFile(path.toURI());
			imageFile.setXMLDocument(getTestXML());
			ImagePlane imagePlane = new ImagePlane(new ImageSeries(imageFile, 0), 0, ImagePlane.ALWAYS_MONOCHROME);
			OMEPlaneMetadataExtractor extractor = new OMEPlaneMetadataExtractor();
			Map<String, String> result = extractor.extract(imagePlane);
			assertNotNull(result);
			assertEquals("Exp1Cam1", result.get("ChannelName"));
			assertEquals("monochrome", result.get("ColorFormat"));
			assertEquals("0", result.get("T"));
			assertEquals("0", result.get("Z"));
			
			imagePlane = new ImagePlane(new ImageSeries(imageFile, 2), 2, ImagePlane.ALWAYS_MONOCHROME);
			result = extractor.extract(imagePlane);
			assertNotNull(result);
			assertEquals("Exp1Cam1", result.get("ChannelName"));
			assertEquals("monochrome", result.get("ColorFormat"));
			assertEquals("0", result.get("T"));
			assertEquals("2", result.get("Z"));
		} catch (ParserConfigurationException e) {
			fail("Unexpected parser configuration exception");
		} catch (SAXException e) {
			fail("Unexpected parsing exception");
		} catch (IOException e) {
			fail("Unexpected I/O exception");
		} catch (DependencyException e) {
			// TODO Auto-generated catch block
			fail("Unexpected dependency exception");
		} catch (ServiceException e) {
			fail("Unexpected service exception");
		}
	}
	
	static InputStream getTestXML() {
		return ClassLoader.getSystemClassLoader().getResourceAsStream("org/cellprofiler/imageset/omexml.xml");
	}
	
	static String getTestXMLAsString() {
		InputStream xmlStream = getTestXML();
		InputStreamReader rdr = new InputStreamReader(xmlStream);
		StringBuffer sb = new StringBuffer();
		CharBuffer cb = CharBuffer.allocate(65536);
		try {
			while(rdr.read(cb) >= 0) {
				sb.append(cb.flip());
				cb.clear();
			}
			return sb.toString();
			
		} catch (IOException e) {
			fail();
			return null; /* never executed */
		}
	}

}
