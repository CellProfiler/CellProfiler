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

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

public class TestWellMetadataExtractor {
	private ImagePlaneDetails makeIPD(String [][] mapKv) {
		final ImagePlaneDetails ipd = Mocks.makeMockIPD();
		for (String []kv: mapKv) ipd.put(kv[0], kv[1]);
		return ipd;
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
