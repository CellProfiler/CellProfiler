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

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.imglib2.meta.TypedAxis;

import org.cellprofiler.imageset.Joiner.JoinerException;
import org.cellprofiler.imageset.filter.Filter;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 *
 */
public class TestJoiner {
	private void assertThrows(List<Joiner.ChannelFilter> channels) {
		try {
			new Joiner(channels, new boolean[channels.get(0).getKeys().length]);
			fail("The joiner did not throw an exception");
		} catch (Joiner.JoinerException e) {
			
		}
	}
	
	private ImagePlaneDetails makeIPD(Joiner.ChannelFilter c, String [] values, String root) {
		String filename = root;
		for (String value:values) filename = filename.concat(value);
		final ImagePlaneDetails result = Mocks.makeMockIPD(filename+".tif");
		addImageSetMetadata(c, result, values);
		return result;
	}

	/**
	 * @param c
	 * @param result
	 * @param values
	 */
	private void addImageSetMetadata(Joiner.ChannelFilter c,
			final Details result, String... values) {
		int [] idxs = c.getJoiningKeyIndices();
		String [] keys = c.getKeys();
		for (int i=0; i<values.length; i++) 
			if (values[i] != null)
				result.put(keys[idxs[i]], values[i]);
	}
	
	private ImagePlaneDetailsStack makeColorStack(Joiner.ChannelFilter c, String [] values, String root) {
		return makeColorStack(c, values, root, "P-12345", "A01", 0);
	}
	
	private ImagePlaneDetailsStack makeColorStack(Joiner.ChannelFilter c, String [] values, String root, String plate, String well, int site) {
		String filename = root;
		for (String value:values) filename = filename.concat(value);
		ImageFileDetails ifd = Mocks.makeMockImageFileDetails(filename, Mocks.MockImageDescription.makeColorDescription(plate, well, site));
		addImageSetMetadata(c, ifd, values);
		final ImagePlaneDetailsStack stack = Mocks.makeMockColorStack(ifd, 3);
		return stack;
	}
	
	private Joiner.ChannelFilter makeMonoChannelFilter(String name, String [] keys, String expression) throws BadFilterExpressionException {
		return makeChannelFilter(name, keys, expression, PlaneStack.XYAxes);
	}
	private Joiner.ChannelFilter makeChannelFilter(String name, String [] keys, String expression, TypedAxis... axes)
	throws BadFilterExpressionException {
		final Filter<ImagePlaneDetailsStack> filter = new Filter<ImagePlaneDetailsStack>(expression, ImagePlaneDetailsStack.class);
		return new Joiner.ChannelFilter(name, keys, filter, axes);
	}
	
	private Joiner.ChannelFilter makeColorChannelFilter(String name, String [] keys, String expression)  throws BadFilterExpressionException {
		return makeChannelFilter(name, keys, expression, PlaneStack.XYCAxes);
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Joiner#Joiner(java.util.List)}.
	 */
	@Test
	public void testJoiner() {
		try {
			String expr = "file does eq \"foo\"";
			Joiner.ChannelFilter a = makeMonoChannelFilter("A", new String [] { "A", "B", "C"}, expr);
			Joiner.ChannelFilter b = makeMonoChannelFilter("B", new String [] { "A", "B", null}, expr);
			Joiner.ChannelFilter c = makeMonoChannelFilter("C", new String [] { "A", null, "C"}, expr);
			Joiner.ChannelFilter d = makeMonoChannelFilter("D", new String [] { "A", "B"}, expr);
			new Joiner(Arrays.asList(a,b), new boolean[3]);
			new Joiner(Arrays.asList(a,b,c), new boolean[3]);
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
			Joiner.ChannelFilter a = makeMonoChannelFilter("A", new String [] { "A" }, "file does contain \"A\"");
			Joiner.ChannelFilter b = makeMonoChannelFilter("B", new String [] { "B" }, "file does contain \"B\"");
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "first"}, "A"),
					makeIPD(a, new String [] { "second"}, "A"),
					makeIPD(b, new String [] { "third" }, "B"),
					makeIPD(b, new String [] { "first" }, "B"),
					makeIPD(a, new String [] { "third" }, "A"),
					makeIPD(b, new String [] { "second" }, "B"));
			Joiner joiner = new Joiner(Arrays.asList(a, b), new boolean[1]);
			Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds, errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0).get(0, 0), ipds.get(0));
			assertSame(foo.get(0).get(1).get(0, 0), ipds.get(3));
			assertSame(foo.get(1).get(0).get(0, 0), ipds.get(1));
			assertSame(foo.get(1).get(1).get(0, 0), ipds.get(5));
			assertSame(foo.get(2).get(0).get(0, 0), ipds.get(4));
			assertSame(foo.get(2).get(1).get(0, 0), ipds.get(2));
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
			Joiner.ChannelFilter a = makeMonoChannelFilter("A", new String [] { "A1", "A2" }, "file does contain \"A\"");
			Joiner.ChannelFilter b = makeMonoChannelFilter("B", new String [] { "B1", "B2" }, "file does contain \"B\"");
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "Plate1", "A01"}, "A"),
					makeIPD(a, new String [] { "Plate1", "A02"}, "A"),
					makeIPD(b, new String [] { "Plate2", "A01" }, "B"),
					makeIPD(b, new String [] { "Plate1", "A01" }, "B"),
					makeIPD(a, new String [] { "Plate2", "A01" }, "A"),
					makeIPD(b, new String [] { "Plate1", "A02" }, "B"));
			Joiner joiner = new Joiner(Arrays.asList(a, b), new boolean[2]);
			Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds, errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0).get(0, 0), ipds.get(0));
			assertSame(foo.get(0).get(1).get(0, 0), ipds.get(3));
			assertSame(foo.get(1).get(0).get(0, 0), ipds.get(1));
			assertSame(foo.get(1).get(1).get(0, 0), ipds.get(5));
			assertSame(foo.get(2).get(0).get(0, 0), ipds.get(4));
			assertSame(foo.get(2).get(1).get(0, 0), ipds.get(2));
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
			Joiner.ChannelFilter a = makeMonoChannelFilter("A", new String [] { "A1", null }, "file does contain \"A\"");
			Joiner.ChannelFilter b = makeMonoChannelFilter("B", new String [] { "B1", "B2" }, "file does contain \"B\"");
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "Plate1"}, "A"),
					makeIPD(b, new String [] { "Plate2", "A01" }, "B"),
					makeIPD(b, new String [] { "Plate1", "A01" }, "B"),
					makeIPD(a, new String [] { "Plate2" }, "A"),
					makeIPD(b, new String [] { "Plate1", "A02" }, "B"));
			Joiner joiner = new Joiner(Arrays.asList(a, b), new boolean[2]);
			Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds, errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0).get(0, 0), ipds.get(0));
			assertSame(foo.get(0).get(1).get(0, 0), ipds.get(2));
			assertSame(foo.get(1).get(0).get(0, 0), ipds.get(0));
			assertSame(foo.get(1).get(1).get(0, 0), ipds.get(4));
			assertSame(foo.get(2).get(0).get(0, 0), ipds.get(3));
			assertSame(foo.get(2).get(1).get(0, 0), ipds.get(1));
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
			Joiner.ChannelFilter a = makeMonoChannelFilter("A", new String [] { "A" }, "file does contain \"A\"");
			Joiner.ChannelFilter b = makeMonoChannelFilter("B", new String [] { "B" }, "file does contain \"B\"");
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "first"}, "A"),
					makeIPD(b, new String [] { "third" }, "B"),
					makeIPD(b, new String [] { "first" }, "B"),
					makeIPD(a, new String [] { "third" }, "A"),
					makeIPD(b, new String [] { "second" }, "B"));
			for (int pass=1; pass <=2; pass++) {
				Joiner joiner = new Joiner((pass == 1)?Arrays.asList(a, b):Arrays.asList(b,a), new boolean[1]);
				Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
				List<ImageSet> foo = joiner.join(ipds, errors );
				assertEquals(2, foo.size());
				int aidx = (pass==1)?0:1;
				int bidx = 1-aidx;
				int resultIdx = 0;
				assertSame(foo.get(resultIdx).get(aidx).get(0, 0),   ipds.get(0));
				assertSame(foo.get(resultIdx++).get(bidx).get(0, 0), ipds.get(2));
				assertSame(foo.get(resultIdx).get(aidx).get(0, 0),   ipds.get(3));
				assertSame(foo.get(resultIdx++).get(bidx).get(0, 0), ipds.get(1));
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
			Joiner.ChannelFilter a = makeMonoChannelFilter("A", new String [] { "A" }, "file does contain \"A\"");
			Joiner.ChannelFilter b = makeMonoChannelFilter("B", new String [] { "B" }, "file does contain \"B\"");
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
				Joiner joiner = new Joiner((pass == 1)?Arrays.asList(a, b):Arrays.asList(b,a), new boolean[1]);
				Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
				List<ImageSet> foo = joiner.join(ipds, errors );
				assertEquals(2, foo.size());
				int aidx = (pass==1)?0:1;
				int bidx = 1-aidx;
				assertSame(foo.get(0).get(aidx).get(0, 0), ipds.get(2));
				assertSame(foo.get(0).get(bidx).get(0, 0), ipds.get(6));
				assertSame(foo.get(1).get(aidx).get(0, 0), ipds.get(5));
				assertSame(foo.get(1).get(bidx).get(0, 0), ipds.get(3));
				assertEquals(1, errors.size());
				ImageSetError error = errors.iterator().next();
				assertTrue(error instanceof ImageSetDuplicateError);
				List<String> key = error.getKey();
				assertEquals(1, key.size());
				assertEquals("first", key.get(0));
				ImageSetDuplicateError e2 = (ImageSetDuplicateError)error;
				List<ImagePlaneDetailsStack> ipdDuplicates = e2.getImagePlaneDetailsStacks();
				assertEquals(3, ipdDuplicates.size());
				List<ImagePlaneDetails> ipdd = new ArrayList<ImagePlaneDetails>();
				for (ImagePlaneDetailsStack s:ipdDuplicates) {
					assertEquals(1, s.getPlaneCount());
					ipdd.add(s.iterator().next());
				}
				assertTrue(ipdd.contains(ipds.get(1)));
				assertTrue(ipdd.contains(ipds.get(4)));
				assertTrue(ipdd.contains(ipds.get(7)));
			}
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (JoinerException e) {
			fail();
		}
	}
	@Test
	public void testColorStack() {
		try {
			Joiner.ChannelFilter a = makeColorChannelFilter("A", new String [] { "A" }, "file does contain \"A\"");
			Joiner.ChannelFilter b = makeColorChannelFilter("B", new String [] { "B" }, "file does contain \"B\"");
			Map<String, ImagePlaneDetailsStack []> stacks = new HashMap<String, ImagePlaneDetailsStack []>();
			stacks.put("first", new ImagePlaneDetailsStack [] { 
				makeColorStack(a, new String [] { "first"}, "A"),
				makeColorStack(b, new String [] { "first" }, "B")});
			stacks.put("second", new ImagePlaneDetailsStack [] {
					makeColorStack(a, new String [] { "second"}, "A"),
					makeColorStack(b, new String [] { "second" }, "B")});
			stacks.put("third", new ImagePlaneDetailsStack [] {
					makeColorStack(a, new String [] { "third" }, "A"),
					makeColorStack(b, new String [] { "third" }, "B")});
			Joiner joiner = new Joiner(Arrays.asList(a, b), new boolean[1]);
			Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImagePlaneDetails> ipds = new ArrayList<ImagePlaneDetails>();
			for (ImagePlaneDetailsStack [] stackks:stacks.values())
				for (ImagePlaneDetailsStack stack:stackks)
					for (ImagePlaneDetails ipd:stack) ipds.add(ipd);
			Collections.shuffle(ipds, new Random(1776));
			List<ImageSet> foo = joiner.join(ipds, errors );
			assertEquals(3, foo.size());
			for (ImageSet imgset:foo) {
				String key = imgset.getKey().get(0);
				assertTrue(stacks.containsKey(key));
				for (int channel=0; channel < 2; channel++) {
					for (int c=0; c<3; c++) {
						assertSame(stacks.get(key)[channel].get(0, 0, c), imgset.get(channel).get(0, 0, c));
					}
				}
			}
		} catch (BadFilterExpressionException e) {
			e.printStackTrace();
			fail();
		} catch (JoinerException e) {
			e.printStackTrace();
			fail();
		}
		
	}
	@Test
	public void testNumericMatching() {
		// Mark the metadata key as numeric. In this case 10 > 2, if it's a string, it's less.
		try {
			Joiner.ChannelFilter a = makeMonoChannelFilter("A", new String [] { "A" }, "file does contain \"A\"");
			Joiner.ChannelFilter b = makeMonoChannelFilter("B", new String [] { "B" }, "file does contain \"B\"");
			List<ImagePlaneDetails> ipds = Arrays.asList(
					makeIPD(a, new String [] { "1"}, "A1"),
					makeIPD(a, new String [] { "2"}, "A2"),
					makeIPD(b, new String [] { "10" }, "B10"),
					makeIPD(b, new String [] { "1" }, "B1"),
					makeIPD(a, new String [] { "10" }, "A10"),
					makeIPD(b, new String [] { "2" }, "B2"));
			Joiner joiner = new Joiner(Arrays.asList(a, b), new boolean [] { true });
			Collection<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> result = joiner.join(ipds, errors);
			assertSame(ipds.get(0), result.get(0).get(0).get(0, 0));
			assertSame(ipds.get(1), result.get(1).get(0).get(0, 0));
			assertSame(ipds.get(2), result.get(2).get(1).get(0, 0));
			assertSame(ipds.get(3), result.get(0).get(1).get(0, 0));
			assertSame(ipds.get(4), result.get(2).get(0).get(0, 0));
			assertSame(ipds.get(5), result.get(1).get(1).get(0, 0));
		} catch (BadFilterExpressionException e) {
			e.printStackTrace();
			fail();
		} catch (JoinerException e) {
			e.printStackTrace();
			fail();
		}
		
	}

}
