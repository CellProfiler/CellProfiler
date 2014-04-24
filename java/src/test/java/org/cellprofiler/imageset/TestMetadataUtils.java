package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.junit.Test;

public class TestMetadataUtils {

	@Test
	public void testGetImageSetMetadata() {
		final ImagePlaneDetailsStack stack00 = Mocks.makeMockMonochromeStack();
		final ImagePlaneDetailsStack stack01 = Mocks.makeMockMonochromeStack();
		final ImagePlaneDetailsStack stack10 = Mocks.makeMockMonochromeStack();
		final ImagePlaneDetailsStack stack11 = Mocks.makeMockMonochromeStack();
		stack00.get(0,0).put("consistent", "foo");
		stack00.get(0,0).put("inconsistent", "bar");
		stack00.get(0,0).put("must_have", "1");
		stack00.get(0,0).put("numeric", "01");
		stack01.get(0,0).put("consistent", "foo");
		stack01.get(0,0).put("inconsistent", "foo");
		stack01.get(0,0).put("must_have", "2");
		stack01.get(0,0).put("numeric", "1");
		stack10.get(0,0).put("consistent", "foo1");
		stack10.get(0,0).put("inconsistent", "bar");
		stack10.get(0,0).put("must_have", "1");
		stack10.get(0,0).put("numeric", "02");
		stack11.get(0,0).put("consistent", "foo1");
		stack11.get(0,0).put("inconsistent", "foo");
		stack11.get(0,0).put("must_have", "2");
		stack11.get(0,0).put("numeric", "2");
		ImageSet imageSet1 = new ImageSet(Arrays.asList(stack00, stack01), Arrays.asList("1"));
		ImageSet imageSet2 = new ImageSet(Arrays.asList(stack10, stack11), Arrays.asList("2"));
		List<ImageSet> imageSets = Arrays.asList(imageSet1, imageSet2);
		Map<String, Integer> mustHave = Collections.singletonMap("must_have", 1);
		Map<String, Comparator<String>> comparators = Collections.singletonMap("numeric", MetadataKeyPair.getNumericComparator());
		Map<String, List<String>> result = MetadataUtils.getImageSetMetadata(imageSets, mustHave, comparators);
		assertEquals(result.get("consistent").get(0), "foo");
		assertEquals(result.get("consistent").get(1), "foo1");
		assertFalse(result.containsKey("inconsistent"));
		assertEquals(result.get("must_have").get(0), "2");
		assertTrue(result.containsKey("numeric"));
	}

	@Test
	public void testCompilePythonRegexp() {
		List<String> keys = new ArrayList<String>();
		Pattern p = MetadataUtils.compilePythonRegexp("(?P<foo>\\d+)-(?P<bar>[a-zA-Z]+)", keys);
		assertEquals("foo", keys.get(0));
		assertEquals("bar", keys.get(1));
		Matcher m = p.matcher("123-abc");
		m.find();
		assertEquals("123", m.group(1));
		assertEquals("abc", m.group(2));
	}

	@Test
	public void testGetIndexPixelsIntIntInt() {
		// TODO: write test to get indexes for all different DimensionOrder
	}

	@Test
	public void testGetIndexPlane() {
		// TODO: write test to get indexes for all different DimensionOrder
	}

}
