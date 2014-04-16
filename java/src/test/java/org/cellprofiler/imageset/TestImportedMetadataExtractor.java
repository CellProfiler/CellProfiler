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
import java.net.URI;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

public class TestImportedMetadataExtractor {
	private ImportedMetadataExtractor makeExtractor(
			String csv, MetadataKeyPair [] keys, boolean expectsFail) {
		StringReader rdr = new StringReader(csv);
		ImportedMetadataExtractor extractor = null;
		try {
			extractor = new ImportedMetadataExtractor(rdr, keys);
			assertFalse(expectsFail);
		} catch (IOException e) {
			assertTrue(expectsFail);
		}
		return extractor;
	}
	
	private void testSomething(ImportedMetadataExtractor extractor, String [][] metadataIn, String [][] expectedOut) {
		ImagePlaneDetails ipd = Mocks.makeMockIPD();
		for (String [] kv:metadataIn) {
			ipd.put(kv[0], kv[1]);
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
				"val1,val2\n", new MetadataKeyPair [] { MetadataKeyPair.makeCaseSensitiveKeyPair("Key1", "Key1") }, false);
		makeExtractor(
				"Key1,Key2\n" +
				"val1,val2\n", new MetadataKeyPair [] { MetadataKeyPair.makeCaseSensitiveKeyPair("Key1", "Key3") }, false);
	}
	
	@Test
	public void testImportedMetadataExtractorMissingKey() {
		makeExtractor(
				"Key1,Key2\n" +
				"val1,val2\n", new MetadataKeyPair [] { MetadataKeyPair.makeCaseSensitiveKeyPair("Key3", "Key1") }, true);
	}
	
	@Test
	public void testDuplicateKey() {
		makeExtractor(
				"Key1,Key1\n" +
				"val1,val2\n", new MetadataKeyPair [] { MetadataKeyPair.makeCaseSensitiveKeyPair("Key1", "Key1") }, true);
	}
	
	@Test
	public void testExtractCaseSensitive() {
		ImportedMetadataExtractor extractor = makeExtractor(
				"Key1,Key2\n" +
				"val1,val2\n" +
				"val3,val4\n", new MetadataKeyPair [] { MetadataKeyPair.makeCaseSensitiveKeyPair("Key1", "Key1a") }, false);
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
				"val3,val4\n", new MetadataKeyPair [] { MetadataKeyPair.makeCaseInsensitiveKeyPair("Key1", "Key1a") }, false);
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
	public void testExtractNumeric() {
		ImportedMetadataExtractor extractor = makeExtractor(
				"Key1,Key2\n" +
				"1,Foo\n" +
				"02,Bar\n", 
				new MetadataKeyPair [] { MetadataKeyPair.makeNumericKeyPair("Key1", "Key1a") },
				false);
		testSomething(extractor, 
				new String [][] { { "Key1a", "01" }}, 
				new String [][] { { "Key2", "Foo"}});
		testSomething(extractor, 
				new String [][] { { "Key1a", "2" }}, 
				new String [][] { { "Key2", "Bar"}});
	}
	
	@Test
	public void testTwoKeys() {
		ImportedMetadataExtractor extractor = makeExtractor(
				"Key1,Key2,Key3,Key4\n" +
				"val11,val21,val31,val41\n" +
				"val12,val22,val32,val42\n", 
				new MetadataKeyPair [] { 
						MetadataKeyPair.makeCaseSensitiveKeyPair("Key1", "Key1"), 
						MetadataKeyPair.makeCaseInsensitiveKeyPair("Key3", "Key3") },
				false);
		testSomething(extractor, 
				new String [][] { {"Key1", "val11"},{"Key3", "val31"}},
				new String [][] { {"Key2", "val21"},{"Key4", "val41"}});
		testSomething(extractor, 
				new String [][] { {"Key1", "val12"},{"Key3", "val32"}},
				new String [][] { {"Key2", "val22"},{"Key4", "val42"}});
		testSomething(extractor, 
				new String [][] { {"Key1", "val12"},{"Key3", "val31"}},
				new String [0][]);
		testSomething(extractor, 
				new String [][] { {"Key1", "VAL12"},{"Key3", "val32"}},
				new String [0][]);
		testSomething(extractor, 
				new String [][] { {"Key1", "val12"},{"Key3", "VAL32"}},
				new String [][] { {"Key2", "val22"},{"Key4", "val42"}});
	}

}
