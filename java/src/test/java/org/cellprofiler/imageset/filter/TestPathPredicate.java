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
package org.cellprofiler.imageset.filter;

import static org.junit.Assert.*;

import java.io.File;
import java.net.URI;
import java.net.URISyntaxException;

import org.cellprofiler.imageset.ImageFile;
import org.cellprofiler.imageset.ImagePlane;
import org.cellprofiler.imageset.ImagePlaneDetails;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.cellprofiler.imageset.filter.TestFileNamePredicate.Expects;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 *
 */
public class TestPathPredicate {

	@Test
	public void testEvalFile() {
		PathPredicate pred = new PathPredicate();
		File root = new File(System.getProperty("user.home"));
		File fileAtPath = new File(root, "foo.jpg");
		String expectedPath = root.getAbsolutePath();
		try {
			pred.setSubpredicates(Expects.expects(expectedPath));
			ImageFile imgfile = new ImageFile(fileAtPath.toURI());
			pred.eval(imgfile);
		} catch (BadFilterExpressionException e) {
			fail("Path predicate takes a subpredicate.");
		}
	}
	@Test
	public void testEvalHTTP() {
		PathPredicate pred = new PathPredicate();
		try {
			pred.setSubpredicates(Expects.expects("http://www.cellprofiler.org/linked_files"));
			ImageFile imgfile = new ImageFile(new URI("http://www.cellprofiler.org/linked_files/bar.jpg"));
			pred.eval(imgfile);
		} catch (BadFilterExpressionException e) {
			fail("Path predicate takes a subpredicate.");
		} catch (URISyntaxException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
