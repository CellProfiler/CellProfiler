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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.net.URI;
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
		ImageFile imageFile = new ImageFile(path.toURI());
		PathNameMetadataExtractor extractor = new PathNameMetadataExtractor(
				new MetadataExtractor<String>() {

			public Map<String, String> extract(String source) {
				assertEquals(source, root.getAbsolutePath());
				return emptyMap;
			}
			public List<String> getMetadataKeys() {
				fail();
				return null;
			}
		});
		extractor.extract(imageFile);
	}
	@Test
	public void testHTTPURL() {
		ImageFile imageFile;
		try {
			imageFile = new ImageFile(new URI("http://cellprofiler.org/linked_files/foo.jpg"));
			PathNameMetadataExtractor extractor = new PathNameMetadataExtractor(
					new MetadataExtractor<String>() {
	
				public Map<String, String> extract(String source) {
					assertEquals(source, "http://cellprofiler.org/linked_files");
					return emptyMap;
				}
				public List<String> getMetadataKeys() {
					fail();
					return null;
				}
			});
			extractor.extract(imageFile);
		} catch (URISyntaxException e) {
			fail();
		}
	}
	@Test
	public void testOMEROURL() {
		ImageFile imageFile;
		try {
			imageFile = new ImageFile(new URI("omero:iid=58038"));
			PathNameMetadataExtractor extractor = new PathNameMetadataExtractor(
					new MetadataExtractor<String>() {
	
				public Map<String, String> extract(String source) {
					assertEquals(source, "");
					return emptyMap;
				}
				public List<String> getMetadataKeys() {
					fail();
					return null;
				}
			});
			extractor.extract(imageFile);
		} catch (URISyntaxException e) {
			fail();
		}
	}
	@Test
	public void testURLWithoutPath() {
		ImageFile imageFile;
		try {
			imageFile = new ImageFile(new URI("http://cellprofiler.org/foo.jpg"));
			PathNameMetadataExtractor extractor = new PathNameMetadataExtractor(
					new MetadataExtractor<String>() {
	
				public Map<String, String> extract(String source) {
					assertEquals(source, "http://cellprofiler.org");
					return emptyMap;
				}
				public List<String> getMetadataKeys() {
					fail();
					return null;
				}
			});
			extractor.extract(imageFile);
		} catch (URISyntaxException e) {
			fail();
		}
	}
}
