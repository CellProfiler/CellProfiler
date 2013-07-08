/**
 * 
 */
package org.cellprofiler.imageset.filter;

import static org.junit.Assert.*;

import org.cellprofiler.imageset.OMEMetadataExtractor;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 * 
 * Test all of the image predicates (color / monochrome / stack / stack frame)
 *
 */
public class TestImagePredicate {

	/**
	 * Test that missing metadata yields an evaluation of false
	 * @param expression the filter expression
	 */
	private void testMissing(String expression) {
		TestFilter.testSomething(expression, "foo.tif", false);
	}
	/**
	 * Test that the IPD with the given metadata key and value
	 * passes the filter
	 * 
	 * @param expression the filter expression
	 * @param key the metadata key
	 * @param value the value for the metadata key
	 */
	private void testPasses(String expression, String key, String value) {
		TestFilter.testSomething(expression, new String [][] {{key, value}}, true);
	}
	/**
	 * Test that the IPD with the given metadata key and value
	 * does not pass the filter
	 * 
	 * @param expression the filter expression
	 * @param key the metadata key
	 * @param value the value for the metadata key
	 */
	private void testDoesNotPass(String expression, String key, String value) {
		TestFilter.testSomething(expression, new String [][] {{key, value}}, false);
	}
	/**
	 * Test method for {@link org.cellprofiler.imageset.filter.ImagePredicate#eval(org.cellprofiler.imageset.filter.ImagePlaneDetails)}.
	 */
	@Test
	public void testMissing() {
		testMissing("image does iscolor");
		testMissing("image does ismonochrome");
		testMissing("image does isstack");
		testMissing("image does isstackframe");
	}
	
	@Test
	public void testPasses() {
		testPasses("image does iscolor", OMEMetadataExtractor.MD_COLOR_FORMAT, OMEMetadataExtractor.MD_RGB);
		testPasses("image does ismonochrome", OMEMetadataExtractor.MD_COLOR_FORMAT, OMEMetadataExtractor.MD_MONOCHROME);
		testPasses("image does isstack", OMEMetadataExtractor.MD_SIZE_T, "3");
		testPasses("image does isstack", OMEMetadataExtractor.MD_SIZE_Z, "3");
		testPasses("image does isstackframe", OMEMetadataExtractor.MD_T, "2");
		testPasses("image does isstackframe", OMEMetadataExtractor.MD_Z, "0");
	}
	@Test
	public void testDoesNotPass() {
		testDoesNotPass("image does iscolor", OMEMetadataExtractor.MD_COLOR_FORMAT, OMEMetadataExtractor.MD_MONOCHROME);
		testDoesNotPass("image does ismonochrome", OMEMetadataExtractor.MD_COLOR_FORMAT, OMEMetadataExtractor.MD_RGB);
		testDoesNotPass("image does isstack", OMEMetadataExtractor.MD_SIZE_T, "1");
		testDoesNotPass("image does isstack", OMEMetadataExtractor.MD_SIZE_Z, "1");
	}
}
