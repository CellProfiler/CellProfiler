package org.cellprofiler.imageset.filter;

import static org.junit.Assert.*;

import java.net.MalformedURLException;
import java.net.URL;
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
			ImageFile imgfile = new ImageFile(new URL("file:///imaging/analysis/foo.jpg"));
			ImagePlaneDetails imgplane = new ImagePlaneDetails(new ImagePlane(imgfile), null);
			pred.eval(imgplane);
		} catch (MalformedURLException e) {
			fail("Oops");
		} catch (BadFilterExpressionException e) {
			// TODO Auto-generated catch block
			fail("File predicate takes a subpredicate");
		}
	}
	@Test
	public void testEvalHTTP() {
		FileNamePredicate pred = new FileNamePredicate();
		try {
			pred.setSubpredicates(Expects.expects("bar.jpg"));
			ImageFile imgfile = new ImageFile(new URL("http://www.cellprofiler.org/linked_files/bar.jpg"));
			ImagePlaneDetails imgplane = new ImagePlaneDetails(new ImagePlane(imgfile), null);
			pred.eval(imgplane);
		} catch (MalformedURLException e) {
			fail("Oops");
		} catch (BadFilterExpressionException e) {
			// TODO Auto-generated catch block
			fail("File predicate takes a subpredicate");
		}
	}
		
}
