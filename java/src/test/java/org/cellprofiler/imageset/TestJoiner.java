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
	//@Test
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
	//@Test
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
			joiner.join(ipds, errors );
		} catch (BadFilterExpressionException e) {
			fail();
		} catch (JoinerException e) {
			fail();
		}
	}

}
