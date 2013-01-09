package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Map;

import org.junit.Test;

public class TestFileNameMetadataExtractor {
	@Test
	public void testFileURL() {
		final String filename = "foo.jpg";
		File path = new File(new File(System.getProperty("user.home")), filename);
		try {
			ImageFile imageFile = new ImageFile(path.toURI().toURL());
			FileNameMetadataExtractor extractor = new FileNameMetadataExtractor(new MetadataExtractor<String>() {

				public Map<String, String> extract(String source) {
					assertEquals(source, filename);
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
			FileNameMetadataExtractor extractor = new FileNameMetadataExtractor(new MetadataExtractor<String>() {

				public Map<String, String> extract(String source) {
					assertEquals(source, "foo.jpg");
					return emptyMap;
				}
			});
			extractor.extract(imageFile);
		} catch (MalformedURLException e) {
			fail();
		}
	}
}
