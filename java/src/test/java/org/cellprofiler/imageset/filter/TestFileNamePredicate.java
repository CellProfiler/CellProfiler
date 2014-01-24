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
package org.cellprofiler.imageset.filter;

import static org.junit.Assert.*;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

import org.cellprofiler.imageset.ImageFile;
import org.cellprofiler.imageset.ImagePlane;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;

public class TestFileNamePredicate {
	static class Expects extends AbstractStringPredicate {
		private final String expected;
		Expects(String expected) {
			this.expected = expected;
		}

		public String getSymbol() {
			return null;
		}

		public void setSubpredicates(List<FilterPredicate<String, ?>> subpredicates) throws BadFilterExpressionException {
		}

		public Class<String> getOutputClass() {
			return String.class;
		}

		@Override
		protected boolean eval(String candidate, String literal) {
			assertEquals(expected, candidate);
			return false;
		}
		static List<FilterPredicate<String, ?>> expects(String name) {
			ArrayList<FilterPredicate<String, ?>> result = new ArrayList<FilterPredicate<String, ?>>();
			result.add(new Expects(name));
			return result;
		}
	};

	@Test
	public void testEvalFile() {
		FileNamePredicate pred = new FileNamePredicate();
		try {
			pred.setSubpredicates(Expects.expects("foo.jpg"));
			ImageFile imgfile = new ImageFile(new URI("file:///imaging/analysis/foo.jpg"));
			ImagePlaneDetails imgplane = new ImagePlaneDetails(new ImagePlane(imgfile), null);
			pred.eval(imgplane);
		} catch (BadFilterExpressionException e) {
			fail("File predicate takes a subpredicate");
		} catch (URISyntaxException e) {
			fail();
		}
	}
	@Test
	public void testEvalHTTP() {
		FileNamePredicate pred = new FileNamePredicate();
		try {
			pred.setSubpredicates(Expects.expects("bar.jpg"));
			ImageFile imgfile = new ImageFile(new URI("http://www.cellprofiler.org/linked_files/bar.jpg"));
			ImagePlaneDetails imgplane = new ImagePlaneDetails(new ImagePlane(imgfile), null);
			pred.eval(imgplane);
		} catch (BadFilterExpressionException e) {
			fail("File predicate takes a subpredicate");
		} catch (URISyntaxException e) {
			fail();
		}
	}
	@Test
	public void testEvalOMERO() {
		FileNamePredicate pred = new FileNamePredicate();
		try {
			pred.setSubpredicates(Expects.expects("iid=12345"));
			ImageFile imgfile = new ImageFile(new URI("omero:iid=12345"));
			ImagePlaneDetails imgplane = new ImagePlaneDetails(new ImagePlane(imgfile), null);
			pred.eval(imgplane);
		} catch (BadFilterExpressionException e) {
			fail("File predicate takes a subpredicate");
		} catch (URISyntaxException e) {
			fail();
		}
		
	}
		
}
