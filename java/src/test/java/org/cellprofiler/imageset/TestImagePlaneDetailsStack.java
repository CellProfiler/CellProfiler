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

import java.io.StringReader;
import java.io.StringWriter;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.TransformerFactoryConfigurationError;
import javax.xml.transform.dom.DOMResult;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;

import net.imglib2.meta.Axes;
import net.imglib2.meta.TypedAxis;

import ome.xml.model.OME;
import ome.xml.model.OMEModel;
import ome.xml.model.OMEModelImpl;
import ome.xml.model.enums.DimensionOrder;
import ome.xml.model.enums.EnumerationException;

import org.junit.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

/**
 * @author Lee Kamentsky
 *
 */
public class TestImagePlaneDetailsStack {

	/**
	 * Test method for {@link org.cellprofiler.imageset.ImagePlaneDetailsStack#ImagePlaneDetailsStack(net.imglib2.meta.TypedAxis[])}.
	 */
	@Test
	public void testImagePlaneDetailsStack() {
		new ImagePlaneDetailsStack(PlaneStack.XYAxes);
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.ImagePlaneDetailsStack#makeMonochromeStack(org.cellprofiler.imageset.ImagePlaneDetails)}.
	 */
	@Test
	public void testMakeMonochromeStack() {
		final ImagePlaneDetails ipd = Mocks.makeMockIPD();
		final ImagePlaneDetailsStack stack = ImagePlaneDetailsStack.makeMonochromeStack(ipd);
		assertEquals(2, stack.numDimensions());
		assertEquals(Axes.X, stack.axis(0).type());
		assertEquals(Axes.Y, stack.axis(1).type());
		assertSame(ipd, stack.get(0,0));
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.ImagePlaneDetailsStack#makeColorStack(org.cellprofiler.imageset.ImagePlaneDetails)}.
	 */
	@Test
	public void testMakeColorStack() {
		final ImagePlaneDetails ipd = Mocks.makeMockIPD();
		final ImagePlaneDetailsStack stack = ImagePlaneDetailsStack.makeColorStack(ipd);
		assertEquals(3, stack.numDimensions());
		assertEquals(Axes.X, stack.axis(0).type());
		assertEquals(Axes.Y, stack.axis(1).type());
		assertEquals(Axes.CHANNEL, stack.axis(2).type());
		assertEquals(1, stack.size(2));
		final ImagePlaneDetails ipdOut = stack.get(0,0,0);
		assertSame(ipd.getImagePlane().getIndex(), ipdOut.getImagePlane().getIndex());
		assertEquals(ImagePlane.ALWAYS_MONOCHROME, ipdOut.getImagePlane().getChannel());
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.ImagePlaneDetailsStack#makeObjectsStack(org.cellprofiler.imageset.ImagePlaneDetails)}.
	 */
	@Test
	public void testMakeObjectsStack() {
		final ImagePlaneDetails ipd = Mocks.makeMockIPD();
		final ImagePlaneDetailsStack stack = ImagePlaneDetailsStack.makeObjectsStack(ipd);
		assertEquals(3, stack.numDimensions());
		assertEquals(Axes.X, stack.axis(0).type());
		assertEquals(Axes.Y, stack.axis(1).type());
		assertEquals(ImagePlaneDetailsStack.OBJECT_PLANE_AXIS_TYPE, stack.axis(2).type());
		assertEquals(1, stack.size(2));
		final ImagePlaneDetails ipdOut = stack.get(0,0,0);
		assertSame(ipd, ipdOut);
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.ImagePlaneDetailsStack#containsKey(java.lang.String)}.
	 */
	@Test
	public void testContainsKey() {
		final ImagePlaneDetails ipd = Mocks.makeMockIPD();
		ipd.put("Foo", "Bar");
		final ImagePlaneDetailsStack stack = ImagePlaneDetailsStack.makeMonochromeStack(ipd);
		assertTrue(stack.containsKey("Foo"));
		assertFalse(stack.containsKey("Bar"));
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.ImagePlaneDetailsStack#get(java.lang.String)}.
	 */
	@Test
	public void testGetString() {
		final ImagePlaneDetails ipd = Mocks.makeMockIPD();
		ipd.put("Foo", "Bar");
		final ImagePlaneDetailsStack stack = ImagePlaneDetailsStack.makeMonochromeStack(ipd);
		assertEquals("Bar", stack.get("Foo"));
		assertNull(stack.get("Bar"));
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.ImagePlaneDetailsStack#addToOME(ome.xml.model.OME, java.lang.String)}.
	 */
	@Test
	public void testOMESingle() {
		final ImagePlaneDetailsStack stackIn = Mocks.makeMockMonochromeStack();
		OME ome = new OME();
		stackIn.addToOME(ome, "Foo");
		checkOMECase(ome, stackIn, "Foo", true);
	}
	
	@Test
	public void testOMEColor() {
		final List<ImagePlaneDetails> ipds = Mocks.makeMockIPDs("foo.jpg", 
				new Mocks.MockImageDescription("P-12345", "A01", 1, 100, 150, 
						3, 1, 1, DimensionOrder.XYCZT, "Red", "Green", "Blue"));
		final ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
		for (ImagePlaneDetails ipd:ipds) {
			stack.add(ipd, 0, 0, stack.size(2));
		}
		OME ome = new OME();
		stack.addToOME(ome, "Foo");
		checkOMECase(ome, stack, "Foo", true);
	}
	@Test
	public void testTwo() {
		final List<ImagePlaneDetails> ipds = Mocks.makeMockIPDs("color.jpg", 
				new Mocks.MockImageDescription("P-12345", "A01", 1, 100, 150, 
						3, 1, 1, DimensionOrder.XYCZT, "Red", "Green", "Blue"));
		final ImagePlaneDetailsStack colorStack = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
		for (ImagePlaneDetails ipd:ipds) {
			colorStack.add(ipd, 0, 0, colorStack.size(2));
		}
		final ImagePlaneDetailsStack monoStack = Mocks.makeMockMonochromeStack();
		OME ome = new OME();
		colorStack.addToOME(ome, "Color");
		monoStack.addToOME(ome, "Mono");
		checkOMECase(ome, colorStack, "Color", true);
		checkOMECase(ome, monoStack, "Mono", false);
	}
	@Test
	public void testWeirdSlices() {
		final Random random = new Random(42789);
		for (DimensionOrder order: new DimensionOrder [] {
				DimensionOrder.XYCTZ, DimensionOrder.XYCZT,
				DimensionOrder.XYTCZ, DimensionOrder.XYTZC,
				DimensionOrder.XYZTC, DimensionOrder.XYZCT
		}) {
			final List<ImagePlaneDetails> ipds = Mocks.makeMockIPDs("foo.jpg", 
					new Mocks.MockImageDescription("P-12345", "A01", 1, 100, 150, 
							3, 5, 10, order, "Red", "Green", "Blue"));
			Collections.shuffle(ipds, random);
			OME ome = new OME();
			final ImagePlaneDetailsStack colorStacks [] = new ImagePlaneDetailsStack[50];
			for (int i=0; i<3*5*10; i+=3) {
				final ImagePlaneDetailsStack colorStack = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
				for (ImagePlaneDetails ipd:ipds) {
					colorStack.add(ipd, 0, 0, colorStack.size(2));
				}
				colorStacks[i/3] = colorStack;
				colorStack.addToOME(ome, String.format("Stack%d", i/3));
			}
			for (int i=0; i<colorStacks.length; i++) {
				checkOMECase(ome, colorStacks[i], String.format("Stack%d", i), i==0);
			}
		}
		
	}
	@Test
	public void testBlank() {
		final List<ImagePlaneDetails> ipds = Mocks.makeMockIPDs("foo.jpg", 
				new Mocks.MockImageDescription("P-12345", "A01", 1, 100, 150, 
						3, 1, 1, DimensionOrder.XYCZT, "Red", "Green", "Blue"));
		final ImagePlaneDetailsStack stack = ImagePlaneDetailsStack.makeColorStack(ipds.get(0));
		stack.add(ipds.get(2), 0,0,2);
		OME ome = new OME();
		stack.addToOME(ome, "Foo");
		final ImagePlaneDetailsStack stackOut = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
		try {
			stackOut.loadFromOME(ome, "Foo");
			ImageFile fileOut = stackOut.get(0, 0, 1).getImagePlane().getImageFile();
			assertTrue(fileOut.isBlank());
			assertEquals(100, fileOut.getBlankSizeX());
			assertEquals(150, fileOut.getBlankSizeY());
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		}
	}
	@Test
	public void testSeries() {
		final List<ImagePlaneDetails> ipds = Mocks.makeMockIPDs("foo.jpg", 
				new Mocks.MockImageDescription("P-12345", "A01", 1, 100, 150, 
						3, 1, 1, DimensionOrder.XYCZT, "Red", "Green", "Blue"),
				new Mocks.MockImageDescription("P-12345", "A02", 1, 100, 150, 
						3, 1, 1, DimensionOrder.XYCZT, "Red", "Green", "Blue"));
		List<ImagePlaneDetailsStack> stacksIn = new ArrayList<ImagePlaneDetailsStack>();
		OME ome = new OME();
		for (int i=0; i<2; i++) {
			final ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
			for (int j=0; j<3; j++) {
				stack.add(ipds.get(i*3+j), 0, 0, j);
			}
			stacksIn.add(stack);
			stack.addToOME(ome, String.format("stack%d", i));
		}
		for (int i=0; i<2; i++) {
			checkOMECase(ome, stacksIn.get(i), String.format("stack%d", i), true);
		}
	}
	@Test
	public void testObjects() {
		final List<ImagePlaneDetails> ipds = Mocks.makeMockIPDs("foo.jpg", 
				new Mocks.MockImageDescription("P-12345", "A01", 1, 100, 150, 
						1, 1, 5, DimensionOrder.XYCZT, "Nuclei"));
		Collections.shuffle(ipds, new Random(1443));
		final ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYOAxes);
		for (int i=0; i< ipds.size(); i++) {
			stack.add(ipds.get(i), 0, 0, i);
		}
		OME ome = new OME();
		stack.addToOME(ome, "Foo");
		checkOMECase(ome, stack, "Foo", true);
	}
	/**
	 * This is a black box test that addToOME and loadFromOME are complimentary
	 * 
	 * @param ome
	 * @param stackIn
	 * @param name
	 */
	private void checkOMECase(OME ome, ImagePlaneDetailsStack stackIn, String name, boolean rwXML) {
		try {
			OME omeOut = ome;
			if (rwXML) {
				Document document = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
				Element omeElement = ome.asXMLElement(document);
				document.appendChild(omeElement);
				DOMSource domSource = new DOMSource(document);
				StringWriter output = new StringWriter();
				final Transformer transformer = TransformerFactory.newInstance().newTransformer();
				transformer.transform(domSource, new StreamResult(output));
				DOMResult domResult = new DOMResult();
				transformer.transform(new StreamSource(new StringReader(output.toString())), domResult);
				Node documentOut = domResult.getNode();
				OMEModel model = new OMEModelImpl();
				omeOut = new OME((Element)(documentOut.getFirstChild()), model);
				model.resolveReferences();
			}
			TypedAxis [] axes = new TypedAxis[stackIn.numDimensions()];
			stackIn.axes(axes);
			ImagePlaneDetailsStack stackOut = new ImagePlaneDetailsStack(axes);
			stackOut.loadFromOME(omeOut, name);
			checkEquals(stackIn, stackOut);
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
			fail();
		} catch (TransformerConfigurationException e) {
			e.printStackTrace();
			fail();
		} catch (TransformerException e) {
			e.printStackTrace();
			fail();
		} catch (TransformerFactoryConfigurationError e) {
			e.printStackTrace();
			fail();
		} catch (EnumerationException e) {
			e.printStackTrace();
			fail();
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		} 
	}

	/**
	 * Check that two stacks are equivalent.
	 * 
	 * @param stackIn
	 * @param stackOut
	 */
	static void checkEquals(ImagePlaneDetailsStack stackIn,
			ImagePlaneDetailsStack stackOut) {
		int nPlanes = 1;
		for (int i=0; i<stackIn.numDimensions(); i++) {
			assertEquals(stackIn.size(i), stackOut.size(i));
			nPlanes *= stackIn.size(i);
		}
		int [] coords = new int[stackIn.numDimensions()];
		for (int i=0; i<nPlanes; i++) {
			final ImagePlane ipIn = stackIn.get(coords).getImagePlane();
			final ImagePlane ipOut = stackOut.get(coords).getImagePlane();
			if (ipIn != null) {
				assertEquals(ipIn.getChannel(), ipOut.getChannel());
				assertEquals(ipIn.getIndex(), ipOut.getIndex());
				assertEquals(ipIn.getSeries().getSeries(), ipOut.getSeries().getSeries());
				assertEquals(ipIn.getImageFile(), ipOut.getImageFile());
			} else {
				assertTrue(ipOut.getImageFile().isBlank());
			}
		}
	}
}
