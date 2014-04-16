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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import net.imglib2.meta.TypedAxis;

import org.cellprofiler.imageset.filter.Filter;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 * Test the Joiner class
 */
public class TestJoiner {
	static private final Comparator<String> I = MetadataKeyPair.getCaseInsensitiveComparator();
	static private final Comparator<String> S = MetadataKeyPair.getCaseSensitiveComparator();
	static private final Comparator<String> N = MetadataKeyPair.getNumericComparator();
	@SuppressWarnings("serial")
	protected class CMKP extends C<MetadataKeyPair> {
		public CMKP(String left, String right, Comparator<String> c) { 
			super(new MetadataKeyPair(left, right, c));
		}
		public CMKP c(String left, String right, Comparator<String> c) {
			add(new MetadataKeyPair(left, right, c));
			return this;
		}
	}
	
	private ImagePlaneDetails makeIPD(String [][] metadata, String root) {
		String filename = root;
		for (String [] kv:metadata) filename = filename.concat(kv[1]);
		final ImagePlaneDetails result = Mocks.makeMockIPD(filename+".tif");
		for (String [] kv:metadata) result.put(kv[0], kv[1]);
		return result;
	}
	private C<ImagePlaneDetails> makeIPDs(String [] keys, String [][] values, String root) {
		C<ImagePlaneDetails> result = new C<ImagePlaneDetails>();
		for (int i=0; i<values.length; i++) {
			final String [][] metadata = new String [keys.length][];
			for (int j=0; j<keys.length; j++) metadata[j] = new String [] {keys[j], values[i][j]};
			result.add(makeIPD(metadata, root));
		}
		return result;
	}

	private ImagePlaneDetailsStack makeColorStack(String [][] metadata, String root) {
		return makeColorStack(metadata, root, "P-12345", "A01", 0);
	}
	
	private ImagePlaneDetailsStack makeColorStack(String [][] metadata, String root, String plate, String well, int site) {
		String filename = root;
		for (String [] kv:metadata) filename = filename.concat(kv[1]);
		ImageFileDetails ifd = Mocks.makeMockImageFileDetails(filename, Mocks.MockImageDescription.makeColorDescription(plate, well, site));
		final ImagePlaneDetailsStack stack = Mocks.makeMockColorStack(ifd, 3);
		for (String [] kv:metadata) ifd.put(kv[0], kv[1]);
		return stack;
	}
	
	private ChannelFilter makeMonoChannelFilter(String name, String expression) throws BadFilterExpressionException {
		return makeChannelFilter(name, expression, PlaneStack.XYAxes);
	}
	private ChannelFilter makeChannelFilter(String name, String expression, TypedAxis... axes)
	throws BadFilterExpressionException {
		final Filter<ImagePlaneDetailsStack> filter = new Filter<ImagePlaneDetailsStack>(expression, ImagePlaneDetailsStack.class);
		return new ChannelFilter(name, filter, axes);
	}
	
	private ChannelFilter makeColorChannelFilter(String name, String expression)  throws BadFilterExpressionException {
		return makeChannelFilter(name, expression, PlaneStack.XYCAxes);
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Joiner#Joiner(java.util.List)}.
	 */
	@Test
	public void testJoiner() {
		try {
			String expr = "file does eq \"foo\"";
			ChannelFilter a = makeMonoChannelFilter("A",  expr);
			new Joiner(a, Arrays.asList(new String [] { "A", "B", "C"}),
					new C<Comparator<String>>(I).c(I).c(I));
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Joiner#join(java.util.List, java.util.Collection)}.
	 */
	@Test
	public void testFullJoin() {
		try {
			final String [] aKeys = new String [] { "A" };
			final String [] bKeys = new String [] { "B" };
			final ChannelFilter a = makeMonoChannelFilter("A", "file does contain \"A\"");
			final ChannelFilter b = makeMonoChannelFilter("B", "file does contain \"B\"");
			final C<ImagePlaneDetails> ipds = 
				makeIPDs(aKeys, new String [][] {{ "first" }, { "second" }, {"third"}}, "A") 
				.c(makeIPDs(bKeys, new String [][] {{ "first" }, { "second" }, {"third"}}, "B"));
			final Joiner joiner = new Joiner(a, Arrays.asList(aKeys), Collections.singletonList(I));
			joiner.addChannel(b, new CMKP("A", "B", I));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds.shuffle(), errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0).get(0, 0), ipds.get(0));
			assertSame(foo.get(0).get(1).get(0, 0), ipds.get(3));
			assertSame(foo.get(1).get(0).get(0, 0), ipds.get(1));
			assertSame(foo.get(1).get(1).get(0, 0), ipds.get(4));
			assertSame(foo.get(2).get(0).get(0, 0), ipds.get(2));
			assertSame(foo.get(2).get(1).get(0, 0), ipds.get(5));
			assertEquals(0, errors.size());
		} catch (BadFilterExpressionException e) {
			fail();
		} 
	}
	@Test
	public void testTwoKeyJoin() {
		try {
			String [] aKeys = new String [] { "A1", "A2" }; 
			String [] bKeys = new String [] { "B1", "B2" };
			ChannelFilter a = makeMonoChannelFilter("A", "file does contain \"A\"");
			ChannelFilter b = makeMonoChannelFilter("B", "file does contain \"B\"");
			C<ImagePlaneDetails> ipds = 
				makeIPDs(aKeys, new String [][] {
						{ "Plate1", "A01"},
						{ "Plate1", "A02"},
						{ "Plate2", "A01" }}, "A")
				.c(makeIPDs(bKeys, new String [][] {
						{ "Plate1", "A01"},
						{ "Plate1", "A02"},
						{ "Plate2", "A01" }}, "B"));
			Joiner joiner = new Joiner(a, Arrays.asList(aKeys), new C<Comparator<String>>(I).c(I));
			joiner.addChannel(b, new CMKP("A1", "B1", I).c("A2", "B2", I));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds.shuffle(), errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0).get(0, 0), ipds.get(0));
			assertSame(foo.get(0).get(1).get(0, 0), ipds.get(3));
			assertSame(foo.get(1).get(0).get(0, 0), ipds.get(1));
			assertSame(foo.get(1).get(1).get(0, 0), ipds.get(4));
			assertSame(foo.get(2).get(0).get(0, 0), ipds.get(2));
			assertSame(foo.get(2).get(1).get(0, 0), ipds.get(5));
			assertEquals(0, errors.size());
		} catch (BadFilterExpressionException e) {
			fail();
		} 	
	}
	@Test
	public void testOuterJoin() {
		try {
			final String [] aKeys = { "A" };
			final String [] bKeys = { "B1", "B2" };
			ChannelFilter a = makeMonoChannelFilter("A", "file does contain \"A\"");
			ChannelFilter b = makeMonoChannelFilter("B", "file does contain \"B\"");
			C<ImagePlaneDetails> ipds = 
				makeIPDs(aKeys, new String [][] {
						{ "Plate1" },
						{ "Plate2" }}, "A")
				.c(makeIPDs(bKeys, new String [][] {
						{ "Plate1", "C01"},
						{ "Plate1", "C02"},
						{ "Plate2", "C01" }}, "B"));
			
			final Joiner joiner = 
				new Joiner(b, Arrays.asList(bKeys), new C<Comparator<String>>(I).c(I));
			joiner.addChannel(a, new CMKP("B1", "A", I));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds.shuffle(), errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0).get(0, 0), ipds.get(2));
			assertSame(foo.get(0).get(1).get(0, 0), ipds.get(0));
			assertSame(foo.get(1).get(0).get(0, 0), ipds.get(3));
			assertSame(foo.get(1).get(1).get(0, 0), ipds.get(0));
			assertSame(foo.get(2).get(0).get(0, 0), ipds.get(4));
			assertSame(foo.get(2).get(1).get(0, 0), ipds.get(1));
			assertEquals(0, errors.size());
		} catch (BadFilterExpressionException e) {
			fail();
		}		
	}
	@Test
	public void testMissing() {
		try {
			final String [] aKeys = new String [] { "A" };
			final String [] bKeys = new String [] { "B" };
			ChannelFilter a = makeMonoChannelFilter("A", "file does contain \"A\"");
			ChannelFilter b = makeMonoChannelFilter("B", "file does contain \"B\"");
			final C<ImagePlaneDetails> ipds = 
				makeIPDs(aKeys, new String [][] {{ "first" }, { "second" }, {"third"}}, "A") 
				.c(makeIPDs(bKeys, new String [][] {{ "first" }, {"third"}}, "B"));
			for (int pass=1; pass <=2; pass++) {
				final ChannelFilter firstCF = (pass==1)?a:b;
				final ChannelFilter secondCF = (pass==1)?b:a;
				final String [] kFirst = (pass==1)?aKeys:bKeys;
				final String [] kSecond = (pass==1)?bKeys:aKeys;
				int aidx = (pass==1)?0:1;
				int bidx = 1-aidx;
				
				Joiner joiner = new Joiner(firstCF, Arrays.asList(kFirst), Collections.singletonList(I));
				joiner.addChannel(secondCF, new CMKP(kFirst[0], kSecond[0], I));
				List<ImageSetError> errors = new ArrayList<ImageSetError>();
				List<ImageSet> foo = joiner.join(ipds.shuffle(), errors );
				assertEquals(2, foo.size());
				int resultIdx = 0;
				assertSame(foo.get(resultIdx).get(aidx).get(0, 0),   ipds.get(0));
				assertSame(foo.get(resultIdx++).get(bidx).get(0, 0), ipds.get(3));
				assertSame(foo.get(resultIdx).get(aidx).get(0, 0),   ipds.get(2));
				assertSame(foo.get(resultIdx++).get(bidx).get(0, 0), ipds.get(4));
				assertEquals(1, errors.size());
				ImageSetError error = errors.iterator().next();
				assertTrue(error instanceof ImageSetMissingError);
				List<String> key = error.getKey();
				assertEquals(1, key.size());
				assertEquals("second", key.get(0));
			}
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
	@Test
	public void testDuplicates() {
		try {
			final String [] aKeys = new String [] { "A" };
			final String [] bKeys = new String [] { "B" };
			final ChannelFilter a = makeMonoChannelFilter("A", "file does contain \"A\"");
			final ChannelFilter b = makeMonoChannelFilter("B", "file does contain \"B\"");
			final C<ImagePlaneDetails> ipds = 
				makeIPDs(aKeys, new String [][] {{ "first" }, { "second" }, {"third"}}, "A") 
				.c(makeIPDs(bKeys, new String [][] {{ "first" }, { "second" }, {"third"}}, "B"))
				.c(makeIPDs(bKeys, new String [][] {{ "first"}}, "B1"));
			for (int pass=1; pass <=2; pass++) {
				final ChannelFilter firstCF = (pass==1)?a:b;
				final ChannelFilter secondCF = (pass==1)?b:a;
				final String [] kFirst = (pass==1)?aKeys:bKeys;
				final String [] kSecond = (pass==1)?bKeys:aKeys;
				int aidx = (pass==1)?0:1;
				int bidx = 1-aidx;
				
				Joiner joiner = new Joiner(firstCF, Arrays.asList(kFirst), Collections.singletonList(I));
				joiner.addChannel(secondCF, new CMKP(kFirst[0], kSecond[0], I));
				List<ImageSetError> errors = new ArrayList<ImageSetError>();
				List<ImageSet> foo = joiner.join(ipds.shuffle(), errors );
				assertSame(foo.get(0).get(aidx).get(0, 0), ipds.get(0));
				assertTrue((foo.get(0).get(bidx).get(0, 0) == ipds.get(6)) ||
						(foo.get(0).get(bidx).get(0, 0) == ipds.get(3)));
				assertSame(foo.get(1).get(aidx).get(0, 0), ipds.get(1));
				assertSame(foo.get(1).get(bidx).get(0, 0), ipds.get(4));
				assertSame(foo.get(2).get(aidx).get(0, 0), ipds.get(2));
				assertSame(foo.get(2).get(bidx).get(0, 0), ipds.get(5));
				assertEquals(1, errors.size());
				ImageSetError error = errors.iterator().next();
				assertTrue(error instanceof ImageSetDuplicateError);
				List<String> key = error.getKey();
				assertEquals(1, key.size());
				assertEquals("first", key.get(0));
				ImageSetDuplicateError e2 = (ImageSetDuplicateError)error;
				List<ImagePlaneDetailsStack> ipdDuplicates = e2.getImagePlaneDetailsStacks();
				assertEquals(1, ipdDuplicates.size());
				List<ImagePlaneDetails> ipdd = new ArrayList<ImagePlaneDetails>();
				for (ImagePlaneDetailsStack s:ipdDuplicates) {
					assertEquals(1, s.getPlaneCount());
					ipdd.add(s.iterator().next());
				}
				assertTrue((ipdd.contains(ipds.get(3))) && (foo.get(0).get(bidx).get(0, 0) == ipds.get(6)) ||
							(ipdd.contains(ipds.get(6))) && (foo.get(0).get(bidx).get(0, 0) == ipds.get(3)));
			}
		} catch (BadFilterExpressionException e) {
			fail();
		}
	}
	@Test
	public void testColorStack() {
		try {
			ChannelFilter a = makeColorChannelFilter("A", "file does contain \"A\"");
			ChannelFilter b = makeColorChannelFilter("B", "file does contain \"B\"");
			Map<String, ImagePlaneDetailsStack []> stacks = new HashMap<String, ImagePlaneDetailsStack []>();
			stacks.put("first", new ImagePlaneDetailsStack [] { 
				makeColorStack(new String [][] { {"A", "first"}}, "A"),
				makeColorStack(new String [][] { {"B", "first" }}, "B")});
			stacks.put("second", new ImagePlaneDetailsStack [] {
					makeColorStack(new String [][] {{ "A", "second" }}, "A"),
					makeColorStack(new String [][] {{ "B", "second" }}, "B")});
			stacks.put("third", new ImagePlaneDetailsStack [] {
					makeColorStack(new String [][] { {"A", "third" }}, "A"),
					makeColorStack(new String [][] { {"B", "third" }}, "B")});
			Joiner joiner = new Joiner(a, Collections.singletonList("A"), Collections.singletonList(I));
			joiner.addChannel(b, new CMKP("A", "B", I));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
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
		} 
		
	}

	@Test
	public void testCaseInsensitiveJoin() {
		try {
			final String [] aKeys = new String [] { "A" };
			final String [] bKeys = new String [] { "B" };
			final ChannelFilter a = makeMonoChannelFilter("A", "file does contain \"A\"");
			final ChannelFilter b = makeMonoChannelFilter("B", "file does contain \"B\"");
			final C<ImagePlaneDetails> ipds = 
				makeIPDs(aKeys, new String [][] {{ "first" }, { "second" }, {"third"}}, "A") 
				.c(makeIPDs(bKeys, new String [][] {{ "fIrst" }, { "Second" }, {"THIRD"}}, "B"));
			final Joiner joiner = new Joiner(a, Arrays.asList(aKeys), Collections.singletonList(I));
			joiner.addChannel(b, new CMKP("A", "B", I));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds.shuffle(), errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0).get(0, 0), ipds.get(0));
			assertSame(foo.get(0).get(1).get(0, 0), ipds.get(3));
			assertSame(foo.get(1).get(0).get(0, 0), ipds.get(1));
			assertSame(foo.get(1).get(1).get(0, 0), ipds.get(4));
			assertSame(foo.get(2).get(0).get(0, 0), ipds.get(2));
			assertSame(foo.get(2).get(1).get(0, 0), ipds.get(5));
			assertEquals(0, errors.size());
		} catch (BadFilterExpressionException e) {
			fail();
		} 
	}
	@Test
	public void testCaseSensitiveJoin() {
		try {
			final String [] aKeys = new String [] { "A" };
			final String [] bKeys = new String [] { "B" };
			final ChannelFilter a = makeMonoChannelFilter("A", "file does contain \"A\"");
			final ChannelFilter b = makeMonoChannelFilter("B", "file does contain \"B\"");
			final C<ImagePlaneDetails> ipds = 
				makeIPDs(aKeys, new String [][] {{"first"}, { "First" }, { "FIRST" }}, "A") 
				.c(makeIPDs(bKeys, new String [][] {{"first"}, { "First" }, { "FIRST" }}, "B"));
			final Joiner joiner = new Joiner(a, Arrays.asList(aKeys), Collections.singletonList(S));
			joiner.addChannel(b, new CMKP("A", "B", S));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> foo = joiner.join(ipds.shuffle(), errors );
			assertEquals(3, foo.size());
			assertSame(foo.get(0).get(0).get(0, 0), ipds.get(0));
			assertSame(foo.get(0).get(1).get(0, 0), ipds.get(3));
			assertSame(foo.get(1).get(0).get(0, 0), ipds.get(1));
			assertSame(foo.get(1).get(1).get(0, 0), ipds.get(4));
			assertSame(foo.get(2).get(0).get(0, 0), ipds.get(2));
			assertSame(foo.get(2).get(1).get(0, 0), ipds.get(5));
			assertEquals(0, errors.size());
		} catch (BadFilterExpressionException e) {
			fail();
		} 
	}
	@Test
	public void testNumericMatching() {
		// Mark the metadata key as numeric. In this case 10 > 2, if it's a string, it's less.
		try {
			ChannelFilter a = makeMonoChannelFilter("A", "file does contain \"A\"");
			ChannelFilter b = makeMonoChannelFilter("B", "file does contain \"B\"");
			C<ImagePlaneDetails> ipds = 
				makeIPDs( new String [] {"A"}, new String [][] {{ "1"}, { "2" }, { "10" }}, "A")
				.c(makeIPDs( new String [] {"B"}, new String [][] {{ "01"}, { "02" }, { "10" }}, "B"));
			Joiner joiner = new Joiner(a, Arrays.asList("A"), Collections.singletonList(N));
			joiner.addChannel(b, Collections.singletonList(MetadataKeyPair.makeNumericKeyPair("A", "B")));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> result = joiner.join(ipds, errors);
			assertSame(ipds.get(0), result.get(0).get(0).get(0, 0));
			assertSame(ipds.get(1), result.get(1).get(0).get(0, 0));
			assertSame(ipds.get(2), result.get(2).get(0).get(0, 0));
			assertSame(ipds.get(3), result.get(0).get(1).get(0, 0));
			assertSame(ipds.get(4), result.get(1).get(1).get(0, 0));
			assertSame(ipds.get(5), result.get(2).get(1).get(0, 0));
		} catch (BadFilterExpressionException e) {
			e.printStackTrace();
			fail();
		}
	}

}
