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

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.xml.parsers.ParserConfigurationException;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;

import ome.xml.model.Image;
import ome.xml.model.OME;
import ome.xml.model.Pixels;

import org.cellprofiler.imageset.filter.Filter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
 * 
 * The extractor operates on image planes but metadata is also extracted
 * from series and files. The extractor maintains a map of ImageFile
 * to ImageFileDetails and ImageSeries to ImageSeriesDetails references.
 * In order to operate efficiently and correctly, you should create a new
 * ImagePlaneMetadataExtractor before processing any image planes and then
 * process all image planes within the same file using the same extractor
 * to utilize the map.
 */
public class ImagePlaneMetadataExtractor  {
	final static public Logger logger = LoggerFactory.getLogger(ImagePlaneMetadataExtractor.class);
	protected class ExtractorFilterPair<T> {
		public final MetadataExtractor<T> extractor;
		public final Filter<ImageFile> filter;
		public final Class<T> klass;
		public ExtractorFilterPair(
				MetadataExtractor<T> extractor, 
				Filter<ImageFile> filter, Class<T> klass) {
			this.extractor = extractor;
			this.filter = filter;
			this.klass = klass;
		}
		/**
		 * Extract the metadata using the extractor, conditional
		 * on the imageFile passing the filter and store in the
		 * destination
		 * 
		 * @param source - run the extractor on the source
		 * @param imageFile - filter based on this image file
		 * @param destination - put the metadata here
		 */
		public void extract(T source, ImageFile imageFile, Details destination) {
			if (filter != null) {
				logger.info(String.format("Running filter %s on %s", filter, source));
				if (! filter.eval(imageFile)) return;
				logger.info("  Filter passed");
			}
			Map<String, String> metadata = extractor.extract(source);
			logger.info(String.format("  Extracted metadata = %s", metadata));
			destination.putAll(metadata);
		}
	}
	final private List<ExtractorFilterPair<ImageFile>> fileExtractors = 
		new ArrayList<ExtractorFilterPair<ImageFile>>();
	
	final private List<ExtractorFilterPair<ImageSeries>> seriesExtractors = 
		new ArrayList<ExtractorFilterPair<ImageSeries>>();
	
	final private List<ExtractorFilterPair<ImagePlane>> planeExtractors =
		new ArrayList<ExtractorFilterPair<ImagePlane>>();
	
	final private List<ExtractorFilterPair<ImagePlaneDetails>> planeDetailsExtractors = 
		new ArrayList<ExtractorFilterPair<ImagePlaneDetails>>();
	
	final private Map<ImageFile, ImageFileDetails> mapImageFileToDetails =
		new IdentityHashMap<ImageFile, ImageFileDetails>();
	
	final private Map<ImageSeries, ImageSeriesDetails> mapImageSeriesToDetails =
		new IdentityHashMap<ImageSeries, ImageSeriesDetails>();
	
	/**
	 * Add a conditional extractor that takes an ImagePlaneDetails and
	 * only operates on files that pass a filter
	 * @param extractor extracts metadata from ImagePlaneDetails
	 * @param filter filter to choose which ImageFiles on which to apply the filter
	 */
	public void addImagePlaneDetailsExtractor(
			MetadataExtractor<ImagePlaneDetails> extractor,
			Filter<ImageFile> filter) {
		planeDetailsExtractors.add(
				new ExtractorFilterPair<ImagePlaneDetails>(
						extractor, filter, ImagePlaneDetails.class));
	}
	/**
	 * Add an unconditional extractor of metadata from ImagePlaneDetails objects
	 * 
	 * @param extractor
	 */
	public void addImagePlaneDetailsExtractor(
			MetadataExtractor<ImagePlaneDetails> extractor) {
		addImagePlaneDetailsExtractor(extractor, null);
	}
	/**
	 * Add a conditional extractor that takes an ImagePlane and only operates on
	 * IPDs that pass a filter
	 *  
	 * @param extractor extract metadata from image planes
	 * @param filter only extract if the ImageFile.
	 */
	public void addImagePlaneExtractor(
			MetadataExtractor<ImagePlane> extractor, 
			Filter<ImageFile> filter) {
		planeExtractors.add(
				new ExtractorFilterPair<ImagePlane>(
						extractor, filter, ImagePlane.class));
	}
	
	/**
	 * Add an unconditional extractor that takes an ImagePlane
	 * 
	 * @param extractor
	 */
	public void addImagePlaneExtractor(MetadataExtractor<ImagePlane> extractor) {
		planeExtractors.add(new ExtractorFilterPair<ImagePlane>(extractor, null, ImagePlane.class));
	}
	
	/**
	 * Add a metadata extractor that extracts metadata from an image file
	 * that passes a given filter
	 * 
	 * @param extractor
	 * @param filter
	 */
	public void addImageFileExtractor(MetadataExtractor<ImageFile> extractor, Filter<ImageFile> filter) {
		fileExtractors.add(new ExtractorFilterPair<ImageFile>(extractor, filter, ImageFile.class));
	}
	
	/**
	 * Add an unconditional image file metadata extractor.
	 * @param extractor
	 */
	public void addImageFileExtractor(MetadataExtractor<ImageFile> extractor) {
		addImageFileExtractor(extractor, null);
	}
	
	/**
	 * Add a metadata extractor that extracts metadata from one of several
	 * series within an image file, filtered using an ImageFile filter.
	 * 
	 * @param extractor
	 * @param filter
	 */
	public void addImageSeriesExtractor(MetadataExtractor<ImageSeries> extractor, Filter<ImageFile> filter) {
		seriesExtractors.add(new ExtractorFilterPair<ImageSeries>(extractor, filter, ImageSeries.class));
	}
	
	/**
	 * Add a metadata extractor that extracts metadata from a series without filtering.
	 * @param extractor
	 */
	public void addImageSeriesExtractor(MetadataExtractor<ImageSeries> extractor) {
		addImageSeriesExtractor(extractor, null);
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
	
	public void addFileNameRegexp(String regexp, Filter<ImageFile> filter) {
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
	
	public void addPathNameRegexp(String regexp, Filter<ImageFile> filter) {
		addImageFileExtractor(new PathNameMetadataExtractor(
				new RegexpMetadataExtractor(regexp)), filter);
	}
	
	/**
	 * Augment the metadata of an ImagePlaneDetails object using
	 * the metadata extracted by this extractor
	 * 
	 * @param details
	 */
	public void extract(ImagePlaneDetails details) {
		for (ExtractorFilterPair<ImagePlaneDetails> efp:planeDetailsExtractors){
			efp.extract(details, details.getImagePlane().getImageFile(), details);
		}
	}
	
	/**
	 * Extract metadata from an image plane.
	 * 
	 * This routine will also calculate series and file metadata
	 * for the plane's ImageSeries and ImageFile (or will
	 * look it up from a previous invocation).
	 *  
	 * @param plane the image plane
	 * @return an ImagePlaneDetails with extracted metadata
	 */
	public ImagePlaneDetails extract(ImagePlane plane) {
		final ImageSeries series = plane.getSeries();
		final ImageFile file = series.getImageFile();
		ImageSeriesDetails seriesDetails = mapImageSeriesToDetails.get(series);
		if (seriesDetails == null) {
			seriesDetails = extract(series);
			mapImageSeriesToDetails.put(series, seriesDetails);
		}
		ImagePlaneDetails ipd = new ImagePlaneDetails(plane, seriesDetails);
		for (ExtractorFilterPair<ImagePlane> efp:planeExtractors) {
			efp.extract(plane, file, ipd);
		}
		for (ExtractorFilterPair<ImagePlaneDetails> efp:planeDetailsExtractors) {
			efp.extract(ipd, file, ipd);
		}
		return ipd;
	}
	
	/**
	 * Extract metadata from an image series.
	 * 
	 * @param series
	 * @return
	 */
	public ImageSeriesDetails extract(ImageSeries series) {
		final ImageFile file = series.getImageFile();
		ImageFileDetails fileDetails = mapImageFileToDetails.get(file);
		if (fileDetails == null) {
			fileDetails = extract(file);
			mapImageFileToDetails.put(file, fileDetails);
		}
		ImageSeriesDetails isd = new ImageSeriesDetails(series, fileDetails);
		for (ExtractorFilterPair<ImageSeries> efp:seriesExtractors) {
			efp.extract(series, file, isd);
		}
		return isd;
	}
	
	/**
	 * Extract metadata from an image file.
	 * 
	 * @param file
	 * @return
	 */
	public ImageFileDetails extract(ImageFile file) {
		ImageFileDetails ifd = new ImageFileDetails(file);
		for (ExtractorFilterPair<ImageFile> efp:fileExtractors) {
			efp.extract(file, file, ifd);
		}
		return ifd;
	}
	
	/**
	 * Extract all image plane details from a list of files
	 * encoded as an array of URL-encoded strings
	 * 
	 * @param urls - the image locations encoded as URLs
	 * @param metadata - the OME XML data for each URL or null if not obtained
	 * @param keysOut - on input, an empty set, on output, the set of all metadata keys extracted.
	 * @return the extracted image plane details.
	 * @throws URISyntaxException 
	 * @throws ServiceException 
	 * @throws DependencyException 
	 * @throws IOException 
	 * @throws SAXException 
	 * @throws ParserConfigurationException 
	 */
	public ImagePlaneDetails [] extract(String [] urls, String [] metadata, Set<String> keysOut) 
	throws URISyntaxException, ParserConfigurationException, SAXException, IOException, DependencyException, ServiceException {
		assert(urls.length == metadata.length);
		final List<ImagePlaneDetails> result = new ArrayList<ImagePlaneDetails> ();
		for (int i=0; i<urls.length; i++) {
			ImageFile imageFile = new ImageFile(new URI(urls[i]));
			if (metadata[i] != null) {
				imageFile.setXMLDocument(metadata[i]);
				final OME fileMetadata = imageFile.getMetadata();
				for (int series=0; series < fileMetadata.sizeOfImageList(); series++) {
					final ImageSeries imageSeries = new ImageSeries(imageFile, series);
					Image seriesMetadata = imageSeries.getOMEImage();
					final Pixels pixels = seriesMetadata.getPixels();
					int nPlanes = pixels.sizeOfPlaneList();
					if (nPlanes == 0) {
						// The planes aren't populated - need to infer from size{C / Z / T}
						nPlanes = pixels.getSizeC().getValue() * pixels.getSizeT().getValue() * pixels.getSizeZ().getValue();
						final ImageSeriesDetails imageSeriesDetails = extract(imageSeries);
						for (int plane=0; plane<nPlanes; plane++) {
							final ImagePlane imagePlane = new ImagePlane(imageSeries, plane, ImagePlane.ALWAYS_MONOCHROME);
							final ImagePlaneDetails ipd = new ImagePlaneDetails(imagePlane, imageSeriesDetails);
							ipd.put(OMEPlaneMetadataExtractor.MD_C, Integer.toString(imagePlane.theC()));
							ipd.put(OMEPlaneMetadataExtractor.MD_T, Integer.toString(imagePlane.theT()));
							ipd.put(OMEPlaneMetadataExtractor.MD_Z, Integer.toString(imagePlane.theZ()));
							result.add(ipd);
						}
						
					} else {
						for (int plane=0; plane<nPlanes; plane++) {
							final ImagePlane imagePlane = new ImagePlane(imageSeries, plane, ImagePlane.ALWAYS_MONOCHROME);
							final ImagePlaneDetails ipd = extract(imagePlane);
							result.add(ipd);
							for (String key:ipd) keysOut.add(key);
						}
					}
				}
			} else {
				// If no OME XML, guess at one plane / one series
				final ImagePlane imagePlane = ImagePlane.makeMonochromePlane(imageFile);
				final ImagePlaneDetails ipd = extract(imagePlane);
				result.add(ipd);
				for (String key:ipd) keysOut.add(key);
			}
		}
		return (ImagePlaneDetails[]) result.toArray(new ImagePlaneDetails[result.size()]);
	}
	/**
	 * @return the metadata keys possibly extracted by the extractors.
	 */
	public List<String> getMetadataKeys() {
		Set<String> keys = new HashSet<String>();
		for (List<ExtractorFilterPair<?>> efs:new List [] {
				this.fileExtractors, this.seriesExtractors, 
				this.planeExtractors, this.planeDetailsExtractors}) {
			for (ExtractorFilterPair<?> ef:efs) {
				keys.addAll(ef.extractor.getMetadataKeys());
			}
		}
		return new ArrayList<String>(keys);
	}
}
