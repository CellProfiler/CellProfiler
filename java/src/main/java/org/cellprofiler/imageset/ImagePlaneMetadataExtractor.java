/**
 * 
 */
package org.cellprofiler.imageset;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;

import org.apache.log4j.Logger;
import org.cellprofiler.imageset.filter.Filter;
import org.cellprofiler.imageset.filter.ImagePlaneDetails;
import org.xml.sax.SAXException;

/**
 * @author Lee Kamentsky
 *
 * The ImagePlaneMetadataExtractor extracts metadata from the OME metadata and
 * from the file URL. Regular expressions can be added to extract metadata
 * from both the path and file name.
 * 
 * The extractor processes the sub-extractors in the order that they were added and,
 * as it does so, it builds up an ImagePlaneDescriptor whose metadata map holds
 * the accumulated metadata. If a sub-extractor was added with a filter, that
 * filter runs on the IPD metadata as accumulated by the previous extractors.
 */
public class ImagePlaneMetadataExtractor  {
	final static public Logger logger = Logger.getLogger(ImagePlaneMetadataExtractor.class);
	protected class ExtractorFilterPair {
		public final MetadataExtractor<ImagePlaneDetails> extractor;
		public final Filter filter;
		public ExtractorFilterPair(
				MetadataExtractor<ImagePlaneDetails> extractor, 
				Filter filter) {
			this.extractor = extractor;
			this.filter = filter;
		}
	}
	final private List<ExtractorFilterPair> extractors = new ArrayList<ExtractorFilterPair>();
	
	/**
	 * Add a conditional extractor that takes an ImagePlaneDetails and only operates on
	 * IPDs that pass a filter
	 *  
	 * @param extractor extract metadata from ipds
	 * @param filter only extract if the ipd passes this filter.
	 */
	public void addImagePlaneDetailsExtractor(MetadataExtractor<ImagePlaneDetails> extractor, Filter filter) {
		extractors.add(new ExtractorFilterPair(extractor, filter));
	}
	
	/**
	 * Add an unconditional extractor that takes an ImagePlaneDetails
	 * 
	 * @param extractor
	 */
	public void addImagePlaneDetailsExtractor(MetadataExtractor<ImagePlaneDetails> extractor) {
		extractors.add(new ExtractorFilterPair(extractor, null));
	}
	
	/**
	 * Add a conditional extractor that takes an image plane
	 * @param extractor extracts metadata from an ImagePlane
	 * @param filter only extract IPDs that pass this filter.
	 */
	public void addImagePlaneExtractor(MetadataExtractor<ImagePlane> extractor, Filter filter) {
		extractors.add(new ExtractorFilterPair(
				new MetadataExtractorAdapter<ImagePlaneDetails, ImagePlane>(extractor) {

					@Override
					protected ImagePlane get(ImagePlaneDetails source) {
						return source.imagePlane;
					}
			}, filter));
	}
	public void addImagePlaneExtractor(MetadataExtractor<ImagePlane> extractor) {
		addImagePlaneExtractor(extractor, null);
	}
	public void addImageFileExtractor(MetadataExtractor<ImageFile> extractor, Filter filter) {
		extractors.add(new ExtractorFilterPair(
				new MetadataExtractorAdapter<ImagePlaneDetails, ImageFile>(extractor) {

					@Override
					protected ImageFile get(ImagePlaneDetails source) {
						return source.imagePlane.getImageFile();
					}
			}, filter));
	}
	public void addImageFileExtractor(MetadataExtractor<ImageFile> extractor) {
		addImageFileExtractor(extractor);
	}
	/**
	 * Add a file name regular expression metadata extractor.
	 * 
	 * @param regexp a Python regular expression with (?P&lt;key&gt;...) syntax
	 *               for extracting regular expressions by key
	 */
	public void addFileNameRegexp(String regexp) {
		addFileNameRegexp(regexp, null);
	}
	
	public void addFileNameRegexp(String regexp, Filter filter) {
		addImageFileExtractor(new FileNameMetadataExtractor(
				new RegexpMetadataExtractor(regexp)), filter);
	}
	
	/**
	 * Add a path name regular expression metadata extractor.
	 * 
	 * @param regexp a Python regular expression with (?P&lt;key&gt;...) syntax
	 *               for extracting regular expressions by key
	 */
	public void addPathNameRegexp(String regexp) {
		addPathNameRegexp(regexp, null);
	}
	
	public void addPathNameRegexp(String regexp, Filter filter) {
		addImageFileExtractor(new PathNameMetadataExtractor(
				new RegexpMetadataExtractor(regexp)), filter);
	}
	
	/**
	 * Extract metadata from an image plane, producing an ImagePlaneDetails
	 * whose metadata is fully populated
	 *  
	 * @param plane the image plane
	 * @return an ImagePlaneDetails with extracted metadata
	 */
	public ImagePlaneDetails extract(ImagePlane plane) {
		ImagePlaneDetails ipd = new ImagePlaneDetails(plane, new HashMap<String, String>());
		for (ExtractorFilterPair efp:extractors) {
			if (efp.filter != null) {
				logger.info(String.format("Running filter %s on %s", efp.filter, plane));
				if (! efp.filter.eval(ipd)) continue;
				logger.info("  Filter passed");
			}
			Map<String, String> metadata = efp.extractor.extract(ipd);
			logger.info(String.format("  Extracted metadata = %s", metadata));
			ipd.metadata.putAll(metadata);
		}
		return ipd;
	}

	/**
	 * The extractMetadata method creates an ImagePlane, extracts metadata from it
	 * and returns an iterator over the metadata map entries. It provides a streamlined
	 * and specialized interface to be used by CellProfiler
	 * 
	 * @param sURL url of image file
	 * @param series series # of image plane
	 * @param index index # of image plane
	 * @param metadata OME xml metadata if present
	 * @param pIPD an array of length 1 that's used to return the Java
	 *        ImagePlaneDetails built by this method, populated with metadata.
	 * @return an iterator over the metadata entries.
	 * @throws IOException 
	 * @throws SAXException 
	 * @throws ParserConfigurationException 
	 */
	public Iterator<Map.Entry<String, String>> extractMetadata(
			String sURL, int series, int index, String metadata, ImagePlaneDetails [] pIPD) 
			throws ParserConfigurationException, SAXException, IOException {
		ImageFile imageFile = new ImageFile(new URL(sURL));
		if (metadata != null)
			imageFile.setXMLDocument(metadata);
		ImagePlane imagePlane = new ImagePlane(imageFile, series, index);
		ImagePlaneDetails result = extract(imagePlane);
		pIPD[0] = result;
		result.imagePlane.getImageFile().clearXMLDocument();
		return result.metadata.entrySet().iterator();
	}
}
