package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.DataFormatException;
import java.util.zip.Deflater;

import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactoryConfigurationError;

import ome.xml.model.enums.EnumerationException;

import org.junit.Test;

public class TestImageSet {

	@Test
	public void testRoundTripNoDictionary() {
		try {
			final ImageSet imageSet = new ImageSet(Arrays.asList(
					Mocks.makeMockColorStack("foo.jpg", 3, "P-12345", "A01", 1),
					Mocks.makeMockColorStack("bar.jpg", 3, "P-12345", "A01", 1)),
					new ArrayList<String>());
			List<String> ids = Arrays.asList("Foo", "Bar");
			final byte [] data = imageSet.compress(ids, null);
			ImageSet output = ImageSet.decompress(data, ids, Arrays.asList(PlaneStack.XYCAxes, PlaneStack.XYCAxes), null);
			checkEquals(imageSet, output);
		} catch (TransformerFactoryConfigurationError e) {
			e.printStackTrace();
			fail();
		} catch (TransformerException e) {
			e.printStackTrace();
			fail();
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		} catch (EnumerationException e) {
			e.printStackTrace();
			fail();
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		} catch (DataFormatException e) {
			e.printStackTrace();
			fail();
		}
	}
	@Test
	public void testRoundTripDictionary() {
		try {
			final List<String> ids = Arrays.asList("Foo", "Bar");
			final ImageSet exemplar = new ImageSet(Arrays.asList(
					Mocks.makeMockColorStack("foo_A01.jpg", 3, "P-12345", "A01", 1),
					Mocks.makeMockColorStack("bar_A01.jpg", 3, "P-12345", "A01", 1)),
					new ArrayList<String>());
			Deflater dNoCompression = new Deflater(Deflater.NO_COMPRESSION);
			byte [] dictionary = exemplar.compress(ids, dNoCompression);
			Deflater dCompression = new Deflater();
			dCompression.setDictionary(dictionary);
			
			final ImageSet imageSet = new ImageSet(Arrays.asList(
					Mocks.makeMockColorStack("foo_A02.jpg", 3, "P-12345", "A02", 1),
					Mocks.makeMockColorStack("bar_A02.jpg", 3, "P-12345", "A02", 1)),
					new ArrayList<String>());
			final byte [] data = imageSet.compress(ids, dCompression);
			ImageSet output = ImageSet.decompress(data, ids, Arrays.asList(PlaneStack.XYCAxes, PlaneStack.XYCAxes), dictionary);
			checkEquals(imageSet, output);
		} catch (TransformerFactoryConfigurationError e) {
			e.printStackTrace();
			fail();
		} catch (TransformerException e) {
			e.printStackTrace();
			fail();
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		} catch (EnumerationException e) {
			e.printStackTrace();
			fail();
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		} catch (DataFormatException e) {
			e.printStackTrace();
			fail();
		}
	}
	@Test
	public void testCreateCompressionDictionary() {
		final List<String> ids = Arrays.asList("Foo", "Bar");
		final List<ImageSet> imageSets = Arrays.asList(
				new ImageSet(Arrays.asList(
						Mocks.makeMockColorStack("foo_A01.jpg", 3, "P-12345", "A01", 1),
						Mocks.makeMockColorStack("bar_A01.jpg", 3, "P-12345", "A01", 1)),
						new ArrayList<String>()),
				new ImageSet(Arrays.asList(
						Mocks.makeMockColorStack("foo_A02.jpg", 3, "P-12345", "A02", 1),
						Mocks.makeMockColorStack("bar_A02.jpg", 3, "P-12345", "A02", 1)),
						new ArrayList<String>()));
		try {
			final byte [][] dataNoDict = {
					imageSets.get(0).compress(ids, new Deflater()),
					imageSets.get(1).compress(ids, new Deflater())
			};
			byte [] dict = ImageSet.createCompressionDictionary(imageSets, ids);
			Deflater deflater = new Deflater();
			deflater.setDictionary(dict);
			final byte [][] dataDict = new byte [2][];
			dataDict[0] = imageSets.get(0).compress(ids, deflater);
			deflater = new Deflater();
			deflater.setDictionary(dict);
			dataDict[1] = imageSets.get(1).compress(ids, deflater);
			assertTrue(dataNoDict[0].length > dataDict[0].length);
			assertTrue(dataNoDict[1].length > dataDict[1].length);
			
		} catch (TransformerFactoryConfigurationError e) {
			e.printStackTrace();
			fail();
		} catch (TransformerException e) {
			e.printStackTrace();
			fail();
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		}
		
	}
	public void checkEquals(ImageSet expected, ImageSet actual) {
		assertEquals(expected.size(), actual.size());
		for (int i=0; i<expected.size(); i++) {
			ImagePlaneDetailsStack stackIn = expected.get(i);
			ImagePlaneDetailsStack stackOut = actual.get(i);
			TestImagePlaneDetailsStack.checkEquals(stackIn, stackOut);
		}
		
	}

}
