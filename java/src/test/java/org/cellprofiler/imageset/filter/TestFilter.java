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
package org.cellprofiler.imageset.filter;

import static org.junit.Assert.*;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import org.cellprofiler.imageset.ImageFile;
import org.cellprofiler.imageset.ImagePlane;
import org.cellprofiler.imageset.OMEMetadataExtractor;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;
import org.junit.BeforeClass;
import org.junit.AfterClass;

/**
 * @author Lee Kamentsky
 *
 */
public class TestFilter {
	static void testSomething(
			String expression, 
			String pathname, 
			String filename, 
			int series,
			int index,
			String [][] metadata, 
			boolean expected) {
		try {
			File rootFile = new File(System.getProperty("user.home"));
			File imagePath = new File(new File(rootFile, pathname), filename);
			ImageFile imageFile;
			imageFile = new ImageFile(imagePath.toURI());
			ImagePlane imagePlane = new ImagePlane(imageFile, series, index);
			Map<String, String> metadataMap = new HashMap<String, String>();
			for (String [] kvpair:metadata) {
				metadataMap.put(kvpair[0], kvpair[1]);
			}
			assertEquals(expected, Filter.filter(expression, new ImagePlaneDetails(imagePlane, metadataMap)));
		} catch (BadFilterExpressionException e) {
			fail(e.getMessage());
		}
	}
	
	static void testSomething(String expression, String pathname, String filename, boolean expected) {
		File file = new File(new File(pathname), filename);
		try {
			assertEquals(expected, Filter.filter(expression, file.toURI()));
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
	
	static void testSomething(String expression, String filename, boolean expected) {
		testSomething(expression, "test", filename, expected);
	}
	
	static void testSomething(String expression, String [][] metadata, boolean expected) {
		testSomething(expression, "test", "foo.jpg", 0, 0, metadata, expected);
	}
	
	static int origCachedEntryCount;
	@BeforeClass
	public static void setUpClass() {
		origCachedEntryCount = Filter.getCachedEntryCount();
		Filter.setCachedEntryCount(4);
	}
	@AfterClass
	public static void tearDownClass() {
		Filter.setCachedEntryCount(origCachedEntryCount);
	}
	@Test
	public void testFilterCache() {
		try {
			for (int i=0; i<20; i++) {
				assertFalse(Filter.filter(String.format("file does contain \"%d\"", i),
						new URI("http://cellprofiler.org/linked_files/foo.jpg")));
			}
			assertTrue(Filter.filter("file does contain \"foo\"",
					new URI("http://cellprofiler.org/linked_files/foo.jpg")));
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (URISyntaxException e) {
			fail();
		}
	}
	@Test
	public void testSimpleExpression() {
		testSomething("file does contain \"foo\"", "foo.jpg", true);
		testSomething("file doesnot contain \"foo\"", "foo.jpg", false);
		testSomething("file does contain \"bar\"", "foo.jpg", false);
	}
	@Test
	public void testCompoundExpression() {
		testSomething("or (file does contain \"foo\") (file does contain \"bar\")", "foo.jpg", true);
		testSomething("and (file does contain \"foo\") (file does contain \"bar\")", "foo.jpg", false);
		testSomething("and (file does contain \"foo\") (file does endwith \"jpg\")", "foo.jpg", true);
	}
	@Test
	public void testNestedExpression() {
		testSomething(
				"or (and (file does contain \"foo\") (file does contain \"bar\")) (and (file does contain \"foo\") (file does endwith \"jpg\"))",
				"foo.jpg", true);
				
		testSomething(
				"and (and (file does contain \"foo\") (file does contain \"bar\")) (and (file does contain \"foo\") (file does endwith \"jpg\"))",
				"foo.jpg", false);
	}
	@Test
	public void testEmptyLiteral() {
		testSomething("file does contain \"\"", "foo.jpg", true);
	}
	@Test
	public void testQuoteEscapedLiteral() {
		testSomething("file does contain \"\\\"\"", "foo\".jpg", true);
		testSomething("file does contain \"\\\"\"", "foo.jpg", false);
		testSomething("file does contain \"\\\"foo\\\".\"", "\"foo\".bar", true);
		testSomething("file does contain \"\\\"foo\\\".\"", "\"foo.bar", false);
		testSomething("file does contain \"\\\"foo\\\".\"", "\"foo\"bar", false);
		testSomething("metadata does SBS\\_Controls \"0\"", new String [][] {{"SBS_Controls", "0"}}, true);
	}
	@Test
	public void testTokenParser() {
		String [][] metadata = {{"foo", "bar"}};
		testSomething("metadata does foo \"bar\"", metadata, true );
		testSomething("metadata does foo \"baz\"", metadata, false);
		testSomething("metadata does baz \"bar\"", metadata, false);
	}
	@Test
	public void testAllPredicateNames() {
		testSomething("file does contain \"foo\"", "foo.jpg", true);
		testSomething("file does startwith \"f\"", "foo.jpg", true);
		testSomething("file does endwith \".jpg\"", "foo.jpg", true);
		testSomething("file does eq \"foo.jpg\"", "foo.jpg", true);
		testSomething("file does containregexp \"oo\\\\.jp\"", "foo.jpg", true);
		testSomething("file does containregexp \"oo\\\\.jp\"", "foo?jpg", false);
		testSomething("file doesnot eq \"foo.jpg\"", "foo.jpg", false);
		testSomething("directory does contain \"foo\"", "foo", "bar.jpg", true);
		testSomething("metadata does foo \"bar\"", new String [][] {{"foo", "bar"}}, true);
		testSomething("metadata doesnot foo \"bar\"", new String [][] {{"foo", "bar"}}, false);
		testSomething("or (file does contain \"foo\")", "foo.jpg", true);
		testSomething("and (file does contain \"foo\")", "foo.jpg", true);
		testSomething("extension does istif", "foo.tif", true);
		testSomething("extension does ispng", "foo.png", true);
		testSomething("extension does isflex", "foo.flex", true);
		testSomething("extension does isimage", "foo.png", true);
		testSomething("extension does isjpeg", "foo.jpg", true);
		testSomething("extension does ismovie", "foo.avi", true);
		testSomething("extension does ispng", "foo.png", true);
		testSomething("extension does istif", "foo.tif", true);
		testSomething("image does iscolor", new String[][] {{OMEMetadataExtractor.MD_COLOR_FORMAT, OMEMetadataExtractor.MD_RGB}}, true);
		testSomething("image does ismonochrome", new String[][] {{OMEMetadataExtractor.MD_COLOR_FORMAT, OMEMetadataExtractor.MD_MONOCHROME}}, true);
		testSomething("image does isstack", new String[][] {{OMEMetadataExtractor.MD_SIZE_T, "3"}}, true);
		testSomething("image does isstackframe", new String [][] {{OMEMetadataExtractor.MD_T, "2"}}, true);
	}
}
