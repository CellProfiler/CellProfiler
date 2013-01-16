/**
 * 
 */
package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.io.File;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cellprofiler.imageset.Joiner.JoinerException;
import org.cellprofiler.imageset.filter.Filter;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.cellprofiler.imageset.filter.ImagePlaneDetails;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 *
 */
public class TestJoiner {
	private void assertThrows(List<Joiner.ChannelFilter> channels) {
		try {
			new Joiner(channels);
			fail("The joiner did not throw an exception");
		} catch (Joiner.JoinerException e) {
			
		}
	}
	
	private ImagePlaneDetails makeIPD(Joiner.ChannelFilter c, String [] values, String root) {
		Map<String, String> metadata = new HashMap<String, String>();
		int [] idxs = c.getJoiningKeyIndices();
		String [] keys = c.getKeys();
		String filename = root;
		for (int i=0; i<values.length; i++) 
			if (values[i] != null)
				metadata.put(keys[idxs[i]], values[i]);
		        filename = filename + values;
		File f = new File(new File(System.getProperty("user.home")), filename);
		try {
			ImageFile imageFile = new ImageFile(f.toURI().toURL());
			return new ImagePlaneDetails(new ImagePlane(imageFile), metadata);
		} catch (MalformedURLException e) {
			fail();
		}
		return null;
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Joiner#Joiner(java.util.List)}.
	 */
	@Test
	public void testJoiner() {
		try {
			Filter filter = new Filter("file does eq \"foo\"");
			Joiner.ChannelFilter a = new Joiner.ChannelFilter("A", new String [] { "A", "B", "C"}, filter);
			Joiner.ChannelFilter b = new Joiner.ChannelFilter("B", new String [] { "A", "B", null}, filter);
			Joiner.ChannelFilter c = new Joiner.ChannelFilter("C", new String [] { "A", null, "C"}, filter);
			Joiner.ChannelFilter d = new Joiner.ChannelFilter("D", new String [] { "A", "B"}, filter);
			new Joiner(Arrays.asList(a,b));
			new Joiner(Arrays.asList(a,b,c));
			assertThrows(Arrays.asList(b,c));
			assertThrows(Arrays.asList(a, d));
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (JoinerException e) {
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Joiner#join(java.util.List, java.util.Collection)}.
	 */
	@Test
	public void testFullJoin() {
		try {
			Joiner.ChannelFilter a = new Joiner.ChannelFilter("A", new String [] { "A" }, new Filter("file does contain \"A\""));
			Joiner.ChannelFilter b = new Joiner.ChannelFilter("B", new String [] { "B" }, new Filter("file does contain \"B\""));
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "first"}, "A"),
					makeIPD(a, new String [] { "second"}, "A"),
					makeIPD(b, new String [] { "third" }, "B"),
					makeIPD(b, new String [] { "first" }, "B"),
					makeIPD(a, new String [] { "third" }, "A"),
					makeIPD(b, new String [] { "second" }, "B"));
			Joiner joiner = new Joiner(Arrays.asList(a, b));
			Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds, errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0), ipds.get(0));
			assertSame(foo.get(0).get(1), ipds.get(3));
			assertSame(foo.get(1).get(0), ipds.get(1));
			assertSame(foo.get(1).get(1), ipds.get(5));
			assertSame(foo.get(2).get(0), ipds.get(4));
			assertSame(foo.get(2).get(1), ipds.get(2));
			assertEquals(0, errors.size());
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (JoinerException e) {
			fail();
		}
	}
	@Test
	public void testTwoKeyJoin() {
		try {
			Joiner.ChannelFilter a = new Joiner.ChannelFilter("A", new String [] { "A1", "A2" }, new Filter("file does contain \"A\""));
			Joiner.ChannelFilter b = new Joiner.ChannelFilter("B", new String [] { "B1", "B2" }, new Filter("file does contain \"B\""));
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "Plate1", "A01"}, "A"),
					makeIPD(a, new String [] { "Plate1", "A02"}, "A"),
					makeIPD(b, new String [] { "Plate2", "A01" }, "B"),
					makeIPD(b, new String [] { "Plate1", "A01" }, "B"),
					makeIPD(a, new String [] { "Plate2", "A01" }, "A"),
					makeIPD(b, new String [] { "Plate1", "A02" }, "B"));
			Joiner joiner = new Joiner(Arrays.asList(a, b));
			Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds, errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0), ipds.get(0));
			assertSame(foo.get(0).get(1), ipds.get(3));
			assertSame(foo.get(1).get(0), ipds.get(1));
			assertSame(foo.get(1).get(1), ipds.get(5));
			assertSame(foo.get(2).get(0), ipds.get(4));
			assertSame(foo.get(2).get(1), ipds.get(2));
			assertEquals(0, errors.size());
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (JoinerException e) {
			fail();
		}		
	}
	@Test
	public void testOuterJoin() {
		try {
			Joiner.ChannelFilter a = new Joiner.ChannelFilter("A", new String [] { "A1", null }, new Filter("file does contain \"A\""));
			Joiner.ChannelFilter b = new Joiner.ChannelFilter("B", new String [] { "B1", "B2" }, new Filter("file does contain \"B\""));
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "Plate1"}, "A"),
					makeIPD(b, new String [] { "Plate2", "A01" }, "B"),
					makeIPD(b, new String [] { "Plate1", "A01" }, "B"),
					makeIPD(a, new String [] { "Plate2" }, "A"),
					makeIPD(b, new String [] { "Plate1", "A02" }, "B"));
			Joiner joiner = new Joiner(Arrays.asList(a, b));
			Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds, errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0), ipds.get(0));
			assertSame(foo.get(0).get(1), ipds.get(2));
			assertSame(foo.get(1).get(0), ipds.get(0));
			assertSame(foo.get(1).get(1), ipds.get(4));
			assertSame(foo.get(2).get(0), ipds.get(3));
			assertSame(foo.get(2).get(1), ipds.get(1));
			assertEquals(0, errors.size());
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (JoinerException e) {
			fail();
		}		
	}
	@Test
	public void testMissing() {
		try {
			Joiner.ChannelFilter a = new Joiner.ChannelFilter("A", new String [] { "A" }, new Filter("file does contain \"A\""));
			Joiner.ChannelFilter b = new Joiner.ChannelFilter("B", new String [] { "B" }, new Filter("file does contain \"B\""));
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "first"}, "A"),
					makeIPD(b, new String [] { "third" }, "B"),
					makeIPD(b, new String [] { "first" }, "B"),
					makeIPD(a, new String [] { "third" }, "A"),
					makeIPD(b, new String [] { "second" }, "B"));
			for (int pass=1; pass <=2; pass++) {
				Joiner joiner = new Joiner((pass == 1)?Arrays.asList(a, b):Arrays.asList(b,a));
				Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
				List<ImageSet> foo = joiner.join(ipds, errors );
				assertEquals(2, foo.size());
				int aidx = (pass==1)?0:1;
				int bidx = 1-aidx;
				int resultIdx = 0;
				assertSame(foo.get(resultIdx).get(aidx), ipds.get(0));
				assertSame(foo.get(resultIdx++).get(bidx), ipds.get(2));
				assertSame(foo.get(resultIdx).get(aidx), ipds.get(3));
				assertSame(foo.get(resultIdx++).get(bidx), ipds.get(1));
				assertEquals(1, errors.size());
				ImageSetError error = errors.iterator().next();
				assertTrue(error instanceof ImageSetMissingError);
				List<String> key = error.getKey();
				assertEquals(1, key.size());
				assertEquals("second", key.get(0));
			}
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (JoinerException e) {
			fail();
		}
	}
	@Test
	public void testDuplicates() {
		try {
			Joiner.ChannelFilter a = new Joiner.ChannelFilter("A", new String [] { "A" }, new Filter("file does contain \"A\""));
			Joiner.ChannelFilter b = new Joiner.ChannelFilter("B", new String [] { "B" }, new Filter("file does contain \"B\""));
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "first"}, "A"),
					makeIPD(b, new String [] { "first" }, "B2"),
					makeIPD(a, new String [] { "second"}, "A"),
					makeIPD(b, new String [] { "third" }, "B"),
					makeIPD(b, new String [] { "first" }, "B1"),
					makeIPD(a, new String [] { "third" }, "A"),
					makeIPD(b, new String [] { "second" }, "B"),
					makeIPD(b, new String [] { "first" }, "B3"));
			for (int pass=1; pass <=2; pass++) {
				Joiner joiner = new Joiner((pass == 1)?Arrays.asList(a, b):Arrays.asList(b,a));
				Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
				List<ImageSet> foo = joiner.join(ipds, errors );
				assertEquals(2, foo.size());
				int aidx = (pass==1)?0:1;
				int bidx = 1-aidx;
				assertSame(foo.get(0).get(aidx), ipds.get(2));
				assertSame(foo.get(0).get(bidx), ipds.get(6));
				assertSame(foo.get(1).get(aidx), ipds.get(5));
				assertSame(foo.get(1).get(bidx), ipds.get(3));
				assertEquals(1, errors.size());
				ImageSetError error = errors.iterator().next();
				assertTrue(error instanceof ImageSetDuplicateError);
				List<String> key = error.getKey();
				assertEquals(1, key.size());
				assertEquals("first", key.get(0));
				ImageSetDuplicateError e2 = (ImageSetDuplicateError)error;
				List<ImagePlaneDetails> ipdDuplicates = e2.getImagePlaneDetails();
				assertEquals(3, ipdDuplicates.size());
				assertTrue(ipdDuplicates.contains(ipds.get(1)));
				assertTrue(ipdDuplicates.contains(ipds.get(4)));
				assertTrue(ipdDuplicates.contains(ipds.get(7)));
			}
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (JoinerException e) {
			fail();
		}
	}

}
