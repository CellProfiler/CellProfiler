package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

import javax.xml.parsers.ParserConfigurationException;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;

import org.junit.Test;
import org.xml.sax.SAXException;

public class TestImageFile {

	@Test
	public void testImageFileURI() {
		URI uri;
		try {
			uri = new URI("http://cellprofiler.org/foo.jpg");
			assertSame(uri, new ImageFile(uri).getURI());
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testImageFileIntInt() {
		ImageFile file = new ImageFile(10, 20);
		assertEquals(ImageFile.BLANK_URI_SCHEME, file.getURI().getScheme());
		assertEquals(ImageFile.BLANK_URI_AUTHORITY, file.getURI().getAuthority());
		assertEquals(ImageFile.BLANK_URI_PATH, file.getURI().getPath());
	}

	@Test
	public void testGetFileName() {
		try {
			assertEquals("foo.jpg", new ImageFile(new URI("http://cellprofiler.org/foo.jpg")).getFileName());
			assertEquals("foo bar.jpg", new ImageFile(new URI("http://cellprofiler.org/foo%20bar.jpg?wtf=1")).getFileName());
			assertEquals("iid=12345", new ImageFile(new URI("omero:iid=12345")).getFileName());
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testGetPathName() {
		try {
			assertEquals("http://cellprofiler.org/foo", new ImageFile(new URI("http://cellprofiler.org/foo/bar.jpg")).getPathName());
			final String home = System.getProperty("user.home");
			final File f = new File(new File(home), "foo.jpg");
			assertEquals(home, new ImageFile(f.toURI()).getPathName());
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testSetXMLDocumentString() {
		try {
			ImageFile imageFile = new ImageFile(new URI("http://cellprofiler.org/foo.jpg"));
			imageFile.setXMLDocument(TestOMEMetadataExtractor.getTestXMLAsString());
			assertEquals(4, imageFile.getMetadata().sizeOfImageList());
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
			fail();
		} catch (SAXException e) {
			e.printStackTrace();
			fail();
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		} catch (DependencyException e) {
			e.printStackTrace();
			fail();
		} catch (ServiceException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testSetXMLDocumentInputStream() {
		try {
			ImageFile imageFile = new ImageFile(new URI("http://cellprofiler.org/foo.jpg"));
			imageFile.setXMLDocument(TestOMEMetadataExtractor.getTestXML());
			assertEquals(4, imageFile.getMetadata().sizeOfImageList());
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
			fail();
		} catch (SAXException e) {
			e.printStackTrace();
			fail();
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		} catch (DependencyException e) {
			e.printStackTrace();
			fail();
		} catch (ServiceException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testClearXMLDocument() {
		try {
			ImageFile imageFile = new ImageFile(new URI("http://cellprofiler.org/foo.jpg"));
			imageFile.setXMLDocument(TestOMEMetadataExtractor.getTestXML());
			imageFile.clearXMLDocument();
			assertNull(imageFile.getMetadata());
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		} catch (ParserConfigurationException e) {
			e.printStackTrace();
			fail();
		} catch (SAXException e) {
			e.printStackTrace();
			fail();
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		} catch (DependencyException e) {
			e.printStackTrace();
			fail();
		} catch (ServiceException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testCompareTo() {
		try {
			ImageFile if1 = new ImageFile(new URI("http://cellprofiler.org/foo.jpg"));
			ImageFile if2 = new ImageFile( new URI("http://cellprofiler.org/bar.jpg"));
			assertEquals(0, if1.compareTo(if1));
			assertTrue(if1.compareTo(if2) > 0);
			assertFalse(if2.compareTo(if1) >= 0);
			assertEquals(0, if1.compareTo(new ImageFile(if1.getURI())));
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testIsBlank() {
		assertTrue(new ImageFile(10, 10).isBlank());
		try {
			assertFalse(new ImageFile(new URI("http://cellprofiler.org/foo.jpg")).isBlank());
		} catch (URISyntaxException e) {
			e.printStackTrace();
			fail();
		}
	}

	@Test
	public void testGetBlankSizeX() {
		assertEquals(10, new ImageFile(10, 20).getBlankSizeX());
	}

	@Test
	public void testGetBlankSizeY() {
		assertEquals(20, new ImageFile(10, 20).getBlankSizeY());
	}

}
