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

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.xml.parsers.ParserConfigurationException;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import loci.common.services.ServiceFactory;
import loci.formats.services.OMEXMLService;

import ome.xml.model.OME;
import ome.xml.model.OMEModelObject;

import org.slf4j.LoggerFactory;
import org.xml.sax.SAXException;

/**
 * @author Lee Kamentsky
 * 
 * ImageFile models an image file. Image files have a URI
 * for their retrieval, and OME metadata retrieved from the file.
 *
 */
public class ImageFile implements Comparable<ImageFile>{
	public static final String BLANK_URI_PATH = "/image/blank";
	public static final String BLANK_URI_AUTHORITY = "cellprofiler.org";
	public static final String BLANK_URI_SCHEME = "info";
	public static final String BLANK_URI_QUERY_PATTERN_STRING = "X=([0-9]+)&Y=([0-9]+)";
	public static final Pattern BLANK_URI_QUERY_PATTERN = Pattern.compile(BLANK_URI_QUERY_PATTERN_STRING);
	private final URI uri;
	private String fileName = null;
	private String pathName = null;
	private OME omexml = null;
	
	/**
	 * Construct an image file from a URL
	 * @param uri - URI to be used to retrieve the file
	 */
	public ImageFile(URI uri) {
		this.uri = uri;
	}
	/**
	 * Construct an image file representing a single plane
	 * of all zeros - a placeholder for a non-existent plane
	 */
	public ImageFile(int xSize, int ySize) {
		String query = String.format("X=%d&Y=%d", xSize, ySize);
		try {
			uri = new URI(BLANK_URI_SCHEME, BLANK_URI_AUTHORITY, BLANK_URI_PATH, query, null);
		} catch (URISyntaxException e) {
			e.printStackTrace();
			throw new AssertionError("Failed to construct the blank image URI");
		}
	}
	
	/**
	 * @return the image file's URI.
	 */
	public URI getURI() {
		return uri;
	}
	
	/**
	 * @return the file name portion of the URL
	 */
	public String getFileName() {
		if (fileName == null) {
			if (uri.getScheme().equals("file")) {
				File file = new File(uri);
				fileName = file.getName();
			} else {
				String path = uri.getPath();
				if (path == null) {
					path = uri.getSchemeSpecificPart();
				}
				int lastslash = path.lastIndexOf("/");
				if (lastslash == -1) {
					fileName = path;
				} else {
					fileName = path.substring(lastslash+1);
				}
			}
		}
		return fileName;
	}
	
	/**
	 * Extract the "path name" from the URL. For file URLs, this is the
	 * path name of the folder containing the file, for others, it is
	 * the portion of the URL preceeding the file name.
	 * 
	 * @return the path name portion of the URL
	 */
	public String getPathName() {
		if (pathName == null) {
			try {
				if (uri.getScheme().equals("file")) {
					final File file = new File(uri);
					pathName = file.getParentFile().getAbsolutePath();
				} else {
					String path = uri.getPath();
					if (path == null) {
						pathName = "";
						return pathName;
					}
					int lastSepIndex = path.lastIndexOf("/");
					if (lastSepIndex <= 0) {
						final URI uriOut = new URI(uri.getScheme(), uri.getHost(), null, null);
						pathName = uriOut.toURL().toString();
					} else {
						final URI uriOut = new URI(
								uri.getScheme(), 
								uri.getHost(), 
								path.substring(0, lastSepIndex),
								null);
						pathName = uriOut.toURL().toString();
					}
				}
			} catch (URISyntaxException e) {
				LoggerFactory.getLogger(getClass()).info(
						"Failed to extract metadata from badly formed URL: " + 
						uri.toString());
				return null;
			} catch (MalformedURLException e) {
				LoggerFactory.getLogger(getClass()).warn(
						String.format("Failed to reconstitute path from URL, \"%s\"", uri.toString()));
				return null;
			}
		}
		return pathName;
	}
	/**
	 * Set the file's OME XML metadata (e.g. as collected by BioFormats)
	 * 
	 * @param omexml - The OME XML as a string
	 * @throws IOException 
	 * @throws SAXException 
	 * @throws ParserConfigurationException 
	 * @throws DependencyException 
	 * @throws ServiceException 
	 */
	public void setXMLDocument(String omexml) throws ParserConfigurationException, SAXException, IOException, DependencyException, ServiceException {
		OMEXMLService svc = new ServiceFactory().getInstance(OMEXMLService.class);
		OMEModelObject root = svc.createOMEXMLRoot(omexml);
		if (! (root instanceof OME))
			throw new ServiceException("Root of XML document wasn't OME");
		this.omexml = (OME)root;
	}

	/**
	 * Set the file's OME XML metadata from a stream
	 * 
	 * @param isOMEXML read the OME XML from this stream
	 * 
	 * @throws ParserConfigurationException
	 * @throws SAXException
	 * @throws IOException
	 * @throws ServiceException 
	 * @throws DependencyException 
	 */
	public void setXMLDocument(InputStream isOMEXML) throws ParserConfigurationException, SAXException, IOException, DependencyException, ServiceException {
		BufferedReader rdr = new BufferedReader(new InputStreamReader(isOMEXML));
		StringBuilder result = new StringBuilder();
		String sep = System.getProperty("line.separator");
		String line;
		while((line=rdr.readLine())!=null) {
			result.append(line);
			result.append(sep);
		}
		setXMLDocument(result.toString());
	}
	
	/**
	 *  Set the file's OME XML metadata using a constructed OME document
	 *  
	 * @param omexml
	 */
	public void setXMLDocument(OME omexml) {
		this.omexml = omexml;
	}
	
	/**
	 * Remove the XML document object (e.g. to save space)
	 */
	public void clearXMLDocument() {
		this.omexml = null;
	}
	
	/**
	 * @return the root of the OME document model.
	 */
	public OME getMetadata() {
		return omexml;
	}
	@Override
	public String toString() {
		return String.format("ImageFile: %s", this.uri);
	}

	/**
	 * Compare two image files lexically by URL
	 * @param o
	 * @return
	 */
	public int compareTo(ImageFile o) {
		if (this == o) return 0;
		return this.uri.compareTo(o.uri);
	}
	/* (non-Javadoc)
	 * @see java.lang.Object#hashCode()
	 */
	@Override
	public int hashCode() {
		return this.uri.hashCode();
	}
	/* (non-Javadoc)
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof ImageFile) return compareTo((ImageFile)obj) == 0;
		return super.equals(obj);
	}
	/**
	 * @return true if the URI of this image file is the URI of a blank (missing) file.
	 */
	public boolean isBlank() {
		return (uri.getScheme().equals(BLANK_URI_SCHEME)) &&
		       (uri.getAuthority().equals(BLANK_URI_AUTHORITY)) &&
		       (uri.getPath().equals(BLANK_URI_PATH));
	}
	/**
	 * For a blank (missing) file, get the plane's X dimension
	 * @return
	 */
	public int getBlankSizeX() {
		Matcher m = BLANK_URI_QUERY_PATTERN.matcher(uri.getQuery());
		m.find();
		return Integer.valueOf(m.group(1));
	}
	/**
	 * For a blank (missing) file, get the plane's Y dimension
	 * @return
	 */
	public int getBlankSizeY() {
		Matcher m = BLANK_URI_QUERY_PATTERN.matcher(uri.getQuery());
		m.find();
		return Integer.valueOf(m.group(2));
	}
}
