package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.util.List;
import java.util.Map;

import ome.xml.model.enums.DimensionOrder;

import org.junit.Test;

public class TestURLSeriesIndexMetadataExtractor {

	@Test
	public void testExtract() {
		final Mocks.MockImageDescription [] descriptions = new Mocks.MockImageDescription [] {
			new Mocks.MockImageDescription("P-12345", "A01", 3, 100, 200, 3, 10, 7, DimensionOrder.XYCZT, "Red", "Green", "Blue")	
		};
		final List<ImagePlaneDetails> ipds = Mocks.makeMockIPDs("foo.jpg", descriptions);
		final URLSeriesIndexMetadataExtractor extractor = new URLSeriesIndexMetadataExtractor();
		for (ImagePlaneDetails ipd:ipds) {
			final ImagePlane imagePlane = ipd.getImagePlane();
			final Map<String, String> metadata = extractor.extract(imagePlane);
			assertEquals(metadata.get(URLSeriesIndexMetadataExtractor.URL_TAG), 
					imagePlane.getImageFile().getURI().toString());
			assertEquals(metadata.get(URLSeriesIndexMetadataExtractor.INDEX_TAG),
					Integer.toString(imagePlane.getIndex()));
			assertEquals(metadata.get(URLSeriesIndexMetadataExtractor.SERIES_TAG),
					Integer.toString(imagePlane.getSeries().getSeries()));
		}
	}

	@Test
	public void testGetMetadataKeys() {
		final List<String> keys = new URLSeriesIndexMetadataExtractor().getMetadataKeys();
		assertTrue(keys.contains(URLSeriesIndexMetadataExtractor.URL_TAG));
		assertTrue(keys.contains(URLSeriesIndexMetadataExtractor.INDEX_TAG));
		assertTrue(keys.contains(URLSeriesIndexMetadataExtractor.SERIES_TAG));
	}

}
