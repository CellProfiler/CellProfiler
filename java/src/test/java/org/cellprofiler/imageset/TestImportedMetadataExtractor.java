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
package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.io.IOException;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Map;

import org.cellprofiler.imageset.ImportedMetadataExtractor.KeyPair;
import org.cellprofiler.imageset.filter.ImagePlaneDetails;
import org.junit.Test;

public class TestImportedMetadataExtractor {
	private ImportedMetadataExtractor makeExtractor(
			String csv, KeyPair [] keys, boolean caseInsensitive, boolean expectsFail) {
		StringReader rdr = new StringReader(csv);
		ImportedMetadataExtractor extractor = null;
		try {
			extractor = new ImportedMetadataExtractor(rdr, keys, caseInsensitive);
			assertFalse(expectsFail);
		} catch (IOException e) {
			assertTrue(expectsFail);
		}
		return extractor;
	}
	
	private void testSomething(ImportedMetadataExtractor extractor, String [][] metadataIn, String [][] expectedOut) {
		ImagePlaneDetails ipd = new ImagePlaneDetails(null, new HashMap<String, String>());
		for (String [] kv:metadataIn) {
			ipd.metadata.put(kv[0], kv[1]);
		}
		Map<String,String> mapOut = extractor.extract(ipd);
		for (String [] kv:expectedOut) {
			assertTrue(mapOut.containsKey(kv[0]));
			assertEquals(kv[1], mapOut.get(kv[0]));
		}
	}

	@Test
	public void testImportedMetadataExtractor() {
		makeExtractor(
				"Key1,Key2\n" +
				"val1,val2\n", new KeyPair [] { new KeyPair("Key1", "Key1") }, false, false);
		makeExtractor(
				"Key1,Key2\n" +
				"val1,val2\n", new KeyPair [] { new KeyPair("Key1", "Key3") }, false, false);
	}
	
	@Test
	public void testImportedMetadataExtractorMissingKey() {
		makeExtractor(
				"Key1,Key2\n" +
				"val1,val2\n", new KeyPair [] { new KeyPair("Key3", "Key1") }, false, true);
	}
	
	@Test
	public void testDuplicateKey() {
		makeExtractor(
				"Key1,Key1\n" +
				"val1,val2\n", new KeyPair [] { new KeyPair("Key1", "Key1") }, false, true);
	}
	
	@Test
	public void testExtractCaseSensitive() {
		ImportedMetadataExtractor extractor = makeExtractor(
				"Key1,Key2\n" +
				"val1,val2\n" +
				"val3,val4\n", new KeyPair [] { new KeyPair("Key1", "Key1a") }, false, false);
		testSomething(extractor, new String [][] { {"Key1a", "val1"}},
				new String [][] { {"Key2", "val2" }});
		testSomething(extractor, new String [][] { {"Key1a", "val3"}},
				new String [][] { {"Key2", "val4" }});
		testSomething(extractor, new String [][] { {"Key1a", "Val1"}},
				new String [0][]);
		
	}
	@Test
	public void testExtractCaseInsensitive() {
		ImportedMetadataExtractor extractor = makeExtractor(
				"Key1,Key2\n" +
				"val1,val2\n" +
				"val3,val4\n", new KeyPair [] { new KeyPair("Key1", "Key1a") }, true, false);
		testSomething(extractor, new String [][] { {"Key1a", "val1"}},
				new String [][] { {"Key2", "val2" }});
		testSomething(extractor, new String [][] { {"Key1a", "val3"}},
				new String [][] { {"Key2", "val4" }});
		testSomething(extractor, new String [][] { {"Key1a", "Val1"}},
				new String [][] { {"Key2", "val2" }});
		testSomething(extractor, new String [][] { {"Key1a", "val5"}},
				new String [0][]);
		testSomething(extractor, new String [][] { {"Key11", "val1"}},
				new String [0][]);
	}
	
	@Test
	public void testTwoKeys() {
		ImportedMetadataExtractor extractor = makeExtractor(
				"Key1,Key2,Key3,Key4\n" +
				"val11,val21,val31,val41\n" +
				"val12,val22,val32,val42\n", 
				new KeyPair [] { new KeyPair("Key1", "Key1"), new KeyPair("Key3", "Key3") },
				false, false);
		testSomething(extractor, 
				new String [][] { {"Key1", "val11"},{"Key3", "val31"}},
				new String [][] { {"Key2", "val21"},{"Key4", "val41"}});
		testSomething(extractor, 
				new String [][] { {"Key1", "val12"},{"Key3", "val32"}},
				new String [][] { {"Key2", "val22"},{"Key4", "val42"}});
		testSomething(extractor, 
				new String [][] { {"Key1", "val12"},{"Key3", "val31"}},
				new String [0][]);
	}

}
