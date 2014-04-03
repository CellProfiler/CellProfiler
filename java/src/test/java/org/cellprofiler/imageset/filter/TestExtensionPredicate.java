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
package org.cellprofiler.imageset.filter;

import static org.junit.Assert.*;

import java.io.File;
import java.util.ArrayList;

import org.cellprofiler.imageset.ImageFile;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 *
 */
public class TestExtensionPredicate {
	private void testSomething(AbstractTerminalPredicate<String> p, String [] pos, String []neg) {
		try {
			ExtensionPredicate ep = new ExtensionPredicate();
			DoesPredicate<String> dp = new DoesPredicate<String>(String.class);
			ArrayList<FilterPredicate<String,?>> a = new ArrayList<FilterPredicate<String,?>>();
			a.add(p);
			dp.setSubpredicates(a);
			a = new ArrayList<FilterPredicate<String, ?>>();
			a.add(dp);
			ep.setSubpredicates(a);
			for (boolean testcase: new boolean [] {true, false}) {
				for (String filename:(testcase? pos:neg)) { 
					File rootFile = new File(System.getProperty("user.home"));
					File testFile = new File(rootFile, filename);
					ImageFile imageFile = new ImageFile(testFile.toURI());
					assertEquals(ep.eval(imageFile), testcase);
				}
			}
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
	@Test
	public void testExtension() {
		testSomething(new IsTifPredicate(),
				new String [] { "foo.tif", "foo.ome.tif" },
				new String [] { "foobar" });
	}
	@Test
	public void testIsFlexPredicate() {
		testSomething(new IsFlexPredicate(), 
				new String [] {"foo.flex", "foo.FLEX"}, 
				new String [] {"foo.jpg"});
	}
	@Test
	public void testIsImagePredicate() {
		testSomething(new IsImagePredicate(),
				new String [] { "foo.tif", "foo.jpg", "foo.png" },
				new String [] { "foo.bar" });
	}
	@Test
	public void testIsJPEGPredicate() {
		testSomething(new IsJPEGPredicate(),
				new String [] { "foo.jpg", "foo.jpeg", "foo.JPG" },
				new String [] { "foo.tif" });
		
	}
	@Test
	public void testIsMoviePredicate() {
		testSomething(new IsMoviePredicate(),
				new String [] { "foo.mov", "foo.avi", "foo.MOV" },
				new String [] { "foo.tif" });
	}
	
	@Test
	public void testIsPNGPredicate() {
		testSomething(new IsPNGPredicate(),
				new String [] { "foo.png", "foo.PNG" },
				new String [] { "foo.jpg" });
	}
	
	@Test
	public void testIsTifPredicate() {
		testSomething(new IsTifPredicate(),
				new String [] { "foo.tif", "foo.tiff", "foo.TiFF" },
				new String [] { "foo.png" });
	}
}
