/**
 * 
 */
package org.cellprofiler.imageset;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Map;

import org.junit.Test;


/**
 * @author Lee Kamentsky
 *
 */
public class TestPathNameMetadataExtractor {
	@Test
	public void testFileURL() {
		final String filename = "foo.jpg";
		final File root = new File(System.getProperty("user.home"));
		File path = new File(root, filename);
		try {
			ImageFile imageFile = new ImageFile(path.toURI().toURL());
			PathNameMetadataExtractor extractor = new PathNameMetadataExtractor(
					new MetadataExtractor<String>() {

				public Map<String, String> extract(String source) {
					assertEquals(source, root.getAbsolutePath());
					return emptyMap;
				}
			});
			extractor.extract(imageFile);
		} catch (MalformedURLException e) {
			fail();
		}
	}
	@Test
	public void testHTTPURL() {
		try {
			ImageFile imageFile = new ImageFile(new URL("http://cellprofiler.org/linked_files/foo.jpg"));
			PathNameMetadataExtractor extractor = new PathNameMetadataExtractor(
					new MetadataExtractor<String>() {

				public Map<String, String> extract(String source) {
					assertEquals(source, "http://cellprofiler.org/linked_files");
					return emptyMap;
				}
			});
			extractor.extract(imageFile);
		} catch (MalformedURLException e) {
			fail();
		}
	}
	@Test
	public void testURLWithoutPath() {
		try {
			ImageFile imageFile = new ImageFile(new URL("http://cellprofiler.org/foo.jpg"));
			PathNameMetadataExtractor extractor = new PathNameMetadataExtractor(
					new MetadataExtractor<String>() {

				public Map<String, String> extract(String source) {
					assertEquals(source, "http://cellprofiler.org");
					return emptyMap;
				}
			});
			extractor.extract(imageFile);
		} catch (MalformedURLException e) {
			fail();
		}
	}
}
