package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;

import org.cellprofiler.imageset.filter.ImagePlaneDetails;
import org.junit.Test;

public class TestWellMetadataExtractor {
	private ImagePlaneDetails makeIPD(String [][] mapKv) {
		Map<String, String> metadata = new HashMap<String, String>();
		for (String []kv: mapKv) metadata.put(kv[0], kv[1]);
		return new ImagePlaneDetails(null, metadata);
	}
	private void testEmptyMapCase(String [][] mapKv) {
		assertEquals(0, new WellMetadataExtractor().extract(makeIPD(mapKv)).size());
	}

	private void testWellCase(String [][] mapKv, String expected) {
		Map<String, String> metadata = new WellMetadataExtractor().extract(makeIPD(mapKv));
		assertEquals(1, metadata.size());
		assertEquals(expected, metadata.get("Well"));
	}
	@Test
	public void testExtract() {
		testEmptyMapCase(new String[0][]);
		testEmptyMapCase(new String[][] {{"row", "A"}});
		testEmptyMapCase(new String[][] {{"col", "10"}});
		testEmptyMapCase(new String[][] {{"Well", "A01"},{"row", "A"}, {"col", "01"}});
		for (String row: new String [] { "wellrow", "well_row", "row" }) {
			for (String col: new String [] { "wellcol", "well_col", "wellcolumn", "well_column",
                    "column", "col" }) {
				testWellCase(new String[][] {{row, "A"}, { col, "01"}}, "A01");
			}
		}
		testWellCase(new String [][] {{ "rOW", "A" }, {"cOl", "01"}}, "A01");
	}

}
