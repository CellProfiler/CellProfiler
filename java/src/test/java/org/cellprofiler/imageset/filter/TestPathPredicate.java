/**
 * 
 */
package org.cellprofiler.imageset.filter;

import static org.junit.Assert.*;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;

import org.cellprofiler.imageset.ImageFile;
import org.cellprofiler.imageset.ImagePlane;
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
			ImageFile imgfile = new ImageFile(fileAtPath.toURI().toURL());
			ImagePlaneDetails imgplane = new ImagePlaneDetails(new ImagePlane(imgfile), null);
			pred.eval(imgplane);
		} catch (MalformedURLException e) {
			fail("Oops");
		} catch (BadFilterExpressionException e) {
			fail("Path predicate takes a subpredicate.");
		}
	}
	@Test
	public void testEvalHTTP() {
		PathPredicate pred = new PathPredicate();
		try {
			pred.setSubpredicates(Expects.expects("http://www.cellprofiler.org/linked_files"));
			ImageFile imgfile = new ImageFile(new URL("http://www.cellprofiler.org/linked_files/bar.jpg"));
			ImagePlaneDetails imgplane = new ImagePlaneDetails(new ImagePlane(imgfile), null);
			pred.eval(imgplane);
		} catch (MalformedURLException e) {
			fail("Oops");
		} catch (BadFilterExpressionException e) {
			fail("Path predicate takes a subpredicate.");
		}
	}
}
