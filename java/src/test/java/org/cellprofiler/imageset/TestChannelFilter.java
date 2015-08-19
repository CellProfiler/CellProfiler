/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
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
import java.util.List;

import ome.xml.model.enums.DimensionOrder;

import org.cellprofiler.imageset.filter.Filter;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;

public class TestChannelFilter {

	@Test
	public void testMakeStacksUnFiltered() {
		final ChannelFilter cf = new ChannelFilter("Foo", PlaneStack.XYAxes);
		List<ImagePlaneDetails> ipds = Arrays.asList(
				Mocks.makeMockIPD("http://cellprofiler.org/foo.jpg"),
				Mocks.makeMockIPD("http://cellprofiler.org/bar.jpg"));
		List<ImagePlaneDetailsStack> stacks = cf.makeStacks(ipds);
		assertEquals(2, stacks.size());
		assertSame(
				ipds.get(0), stacks.get(0).get(0, 0));
		assertEquals(
				ipds.get(1), stacks.get(1).get(0, 0));
	}
	@Test
	public void testMakeStacksFiltered() {
		try {
			final Filter<ImagePlaneDetailsStack> f = 
				new Filter<ImagePlaneDetailsStack>("file does contain \"foo\"", ImagePlaneDetailsStack.class);
			final ChannelFilter cf = new ChannelFilter("Foo", f, PlaneStack.XYAxes);
			List<ImagePlaneDetails> ipds = Arrays.asList(
					Mocks.makeMockIPD("http://cellprofiler.org/foo.jpg"),
					Mocks.makeMockIPD("http://cellprofiler.org/bar.jpg"));
			List<ImagePlaneDetailsStack> stacks = cf.makeStacks(ipds);
			assertEquals(1, stacks.size());
			assertSame(
					ipds.get(0), stacks.get(0).get(0, 0));
		} catch (BadFilterExpressionException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testMakeImageSets() {
		try {
			final Filter<ImagePlaneDetailsStack> fFoo = 
				new Filter<ImagePlaneDetailsStack>("file does contain \"foo\"", ImagePlaneDetailsStack.class);
			final ChannelFilter cfFoo = new ChannelFilter("Foo", fFoo, PlaneStack.XYAxes);
			final Filter<ImagePlaneDetailsStack> fBar = 
				new Filter<ImagePlaneDetailsStack>("file does contain \"bar\"", ImagePlaneDetailsStack.class);
			final ChannelFilter cfBar = new ChannelFilter("Bar", fBar, PlaneStack.XYAxes);
			List<ImagePlaneDetails> ipds = Arrays.asList(
					Mocks.makeMockIPD("http://cellprofiler.org/1/bar.jpg"),
					Mocks.makeMockIPD("http://cellprofiler.org/2/foo.jpg"),
					Mocks.makeMockIPD("http://cellprofiler.org/1/foo.jpg"),
					Mocks.makeMockIPD("http://cellprofiler.org/2/bar.jpg"));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> imageSets = ChannelFilter.makeImageSets(Arrays.asList(cfFoo, cfBar), ipds, errors);
			assertEquals(2, imageSets.size());
			assertEquals(0, errors.size());
			ImageSet imageSet = imageSets.get(0);
			assertEquals(2, imageSet.size());
			assertSame(ipds.get(2), imageSet.get(0).get(0,0));
			assertSame(ipds.get(0), imageSet.get(1).get(0,0));
			assertEquals("1", imageSet.getKey().get(0));
			imageSet = imageSets.get(1);
			assertEquals(2, imageSet.size());
			assertSame(ipds.get(1), imageSet.get(0).get(0,0));
			assertSame(ipds.get(3), imageSet.get(1).get(0,0));
			assertEquals("2", imageSet.getKey().get(0));		
		} catch (BadFilterExpressionException e) {
			e.printStackTrace();
			fail();
		}
	}
	@Test
	public void testMakeImageSetsWithMissing() {
		try {
			final Filter<ImagePlaneDetailsStack> fFoo = 
				new Filter<ImagePlaneDetailsStack>("file does contain \"foo\"", ImagePlaneDetailsStack.class);
			final ChannelFilter cfFoo = new ChannelFilter("Foo", fFoo, PlaneStack.XYAxes);
			final Filter<ImagePlaneDetailsStack> fBar = 
				new Filter<ImagePlaneDetailsStack>("file does contain \"bar\"", ImagePlaneDetailsStack.class);
			final ChannelFilter cfBar = new ChannelFilter("Bar", fBar, PlaneStack.XYAxes);
			List<ImagePlaneDetails> ipds = Arrays.asList(
					Mocks.makeMockIPD("http://cellprofiler.org/1/bar.jpg"),
					Mocks.makeMockIPD("http://cellprofiler.org/2/foo.jpg"),
					Mocks.makeMockIPD("http://cellprofiler.org/1/foo.jpg"),
					Mocks.makeMockIPD("http://cellprofiler.org/2/bar.jpg"),
					Mocks.makeMockIPD("http://cellprofiler.org/3/bar.jpg"));
			List<ImageSetError> errors = new ArrayList<ImageSetError>();
			List<ImageSet> imageSets = ChannelFilter.makeImageSets(Arrays.asList(cfFoo, cfBar), ipds, errors);
			assertEquals(2, imageSets.size());
			ImageSet imageSet = imageSets.get(0);
			assertEquals(2, imageSet.size());
			assertSame(ipds.get(2), imageSet.get(0).get(0,0));
			assertSame(ipds.get(0), imageSet.get(1).get(0,0));
			assertEquals("1", imageSet.getKey().get(0));
			imageSet = imageSets.get(1);
			assertEquals(2, imageSet.size());
			assertSame(ipds.get(1), imageSet.get(0).get(0,0));
			assertSame(ipds.get(3), imageSet.get(1).get(0,0));
			assertEquals("2", imageSet.getKey().get(0));
			assertEquals(1, errors.size());
			ImageSetMissingError error = (ImageSetMissingError)errors.get(0);
			assertEquals("Foo", error.getChannelName());
			assertEquals("3", error.getKey().get(0));
			
		} catch (BadFilterExpressionException e) {
			e.printStackTrace();
			fail();
		}
	}
	@Test
	public void testMakeObjectsStacksSinglePlanes() {
		C<ImagePlaneDetails> ipds = new C<ImagePlaneDetails>(
				Mocks.makeMockIPD("http://cellprofiler.org/foo.jpg"))
				.c(Mocks.makeMockIPD("http://cellprofiler.org/bar.jpg"));
		ChannelFilter cf = new ChannelFilter("Foo", PlaneStack.XYOAxes);
		List<ImagePlaneDetailsStack> stacks = cf.makeStacks(ipds);
		assertEquals(2, stacks.size());
		assertSame(stacks.get(0).get(0,0,0).getImagePlane().getImageFile(),
				   ipds.get(1).getImagePlane().getImageFile());
		assertSame(stacks.get(1).get(0,0,0).getImagePlane().getImageFile(),
				   ipds.get(0).getImagePlane().getImageFile());
	}
	@Test
	public void testMakeObjectStacksMultiSeries() {
		C<ImagePlaneDetails> ipds = new C<ImagePlaneDetails>(
				Mocks.makeMockIPDs("http://cellprofiler.org/foo.jpg", 
						new Mocks.MockImageDescription("P-12345", "A01", 1, 100, 200, 1, 3, 4, DimensionOrder.XYCTZ, "labels"),
						new Mocks.MockImageDescription("P-12345", "A02", 1, 100, 200, 1, 8, 2, DimensionOrder.XYCZT, "labels")));
		ChannelFilter cf = new ChannelFilter("Foo", PlaneStack.XYOAxes);
		List<ImagePlaneDetailsStack> stacks = cf.makeStacks(ipds.shuffle());
		assertEquals(2, stacks.size());
		for (ImagePlaneDetailsStack stack:stacks) {
			assertEquals(3, stack.numDimensions());
			assertSame(PlaneStack.OBJECT_PLANE_AXIS_TYPE, stack.axis(2).type());
		}
		assertEquals(12, stacks.get(0).size(2));
		for (int i=0; i<12; i++) {
			final ImagePlane plane = stacks.get(0).get(0, 0, i).getImagePlane();
			assertEquals(i/4, plane.getOMEPlane().getTheZ().getValue().intValue());
			assertEquals(i%4, plane.getOMEPlane().getTheT().getValue().intValue());
			assertEquals(i, plane.getIndex());
		}
		assertEquals(16, stacks.get(1).size(2));
		for (int i=0; i<16; i++) {
			final ImagePlane plane = stacks.get(1).get(0, 0, i).getImagePlane();
			assertEquals(i % 8, plane.getOMEPlane().getTheZ().getValue().intValue());
			assertEquals(i / 8, plane.getOMEPlane().getTheT().getValue().intValue());
			assertEquals(i, plane.getIndex());
		}
	}
}
