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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.logging.Logger;
import java.util.zip.DataFormatException;
import java.util.zip.Deflater;
import java.util.zip.DeflaterInputStream;
import java.util.zip.DeflaterOutputStream;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;

import javax.xml.parsers.DocumentBuilder;
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

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

import net.imglib2.meta.TypedAxis;

import ome.xml.model.OME;
import ome.xml.model.OMEModel;
import ome.xml.model.OMEModelImpl;
import ome.xml.model.enums.EnumerationException;


/**
 * @author Lee Kamentsky
 *
 * An ImageSet is a collection of image stacks coallated for
 * processing during a CellProfiler cycle.
 */
public class ImageSet extends ArrayList<ImagePlaneDetailsStack> {
	private static final long serialVersionUID = -6824821112413090930L;
	private static final Logger logger = Logger.getLogger(ImageSet.class.getCanonicalName());
	final private List<String> key;
	/**
	 * Construct the image set from its image plane descriptors and key
	 * @param ipds
	 * @param key
	 */
	public ImageSet(Collection<ImagePlaneDetailsStack> ipds, List<String> key) {
		super(ipds);
		this.key = key;
	}
	
	/**
	 * A convenience constructor if you want to start out with just a single
	 * channel and then glom on.
	 * 
	 * @param stack the stack for the first channel in the image set
	 * @param key the key that defines the image set.
	 */
	public ImageSet(ImagePlaneDetailsStack stack, List<String> key) {
		this(icantfindafunctionthatmakesamutablelistinitializedwithasinglemember(stack), key);
	}
	
	public List<String> getKey() {
		return key;
	}
	
	/**
	 * Build OME-XML images for each stack in the image set.
	 * 
	 * @param ome - OME xml root node
	 * @param ids - ids for the image elements for each of the images
	 */
	public void addToOME(OME ome, List<String> ids) {
		//TODO: handle X/Y tiling.
		for (int i=0; i<ids.size(); i++) {
			final ImagePlaneDetailsStack ipds = get(i);
			final String id = ids.get(i);
			ipds.addToOME(ome, id);
		}
	}

	private static ThreadLocal<Transformer> transformer = new ThreadLocal<Transformer>() {

		/* (non-Javadoc)
		 * @see java.lang.ThreadLocal#initialValue()
		 */
		@Override
		protected Transformer initialValue() {
			try {
				return TransformerFactory.newInstance().newTransformer();
			} catch (TransformerConfigurationException e) {
				logger.severe(String.format("Failed to create the ImageSet XML transformer: %s", e.getMessage()) );
			} catch (TransformerFactoryConfigurationError e) {
				logger.severe(String.format("Failed to create the ImageSet XML transformer: %s", e.getMessage()) );
			}
			return null;
		}
	};
	private static ThreadLocal<DocumentBuilder> documentBuilder = new ThreadLocal<DocumentBuilder> () {
		/* (non-Javadoc)
		 * @see java.lang.ThreadLocal#initialValue()
		 */
		@Override
		protected DocumentBuilder initialValue() {
			try {
				return DocumentBuilderFactory.newInstance().newDocumentBuilder();
			} catch (ParserConfigurationException e) {
				logger.severe(String.format("Failed to create an XML document builder: %s", e.getMessage()) );
			}
			return null;
		}		
	};
	
	/**
	 * Compress the OME-XML representation of the image set into a gzipped blob
	 * @param ids the names of the channels
	 * @param deflater to use to perform the compression or null to use the default.
	 * @return the compressed representation.
	 * @throws TransformerFactoryConfigurationError 
	 * @throws TransformerException 
	 * @throws IOException 
	 */
	public byte [] compress(List<String> ids, Deflater deflater) 
	throws TransformerFactoryConfigurationError, TransformerException, IOException {
		OME ome = new OME();
		addToOME(ome, ids);
		Document document = documentBuilder.get().newDocument();
		Element omeElement = ome.asXMLElement(document);
		document.appendChild(omeElement);
		DOMSource domSource = new DOMSource(document);
		
		ByteArrayOutputStream baOS = new ByteArrayOutputStream();
		OutputStream os = (deflater == null)?new DeflaterOutputStream(baOS):new DeflaterOutputStream(baOS, deflater);
		final Transformer transformer = TransformerFactory.newInstance().newTransformer();
		final OutputStreamWriter writer = new OutputStreamWriter(os);
		transformer.transform(domSource, new StreamResult(writer));
		writer.close();
		os.close();
		return baOS.toByteArray();
	}
	
	/**
	 * @param data - the data previously compressed by compress
	 * @param ids - the names of the channels
	 * @param axesList - one axes description per channel
	 * @param dictionary - a dictionary to be used to prime the inflater when decompressing.
	 * @return the image set.
	 * @throws EnumerationException if the OMEModel failed to be instantiated due to misconfiguration.
	 * @throws TransformerException if the XML is not properly formed
	 * @throws URISyntaxException if the URI of one of the image files is not properly constructed in the XML
	 * @throws DataFormatException 
	 */
	static public ImageSet decompress(byte [] data, List<String> ids, List<TypedAxis []> axesList, byte [] dictionary) 
		throws EnumerationException, TransformerException, URISyntaxException, IOException, DataFormatException {
		DOMResult domResult = new DOMResult();
		//
		// First, decompress the data using a ByteOutputStream to coallate
		//
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		Inflater inflater = new Inflater();
		inflater.setInput(data);
		byte [] buffer = new byte[4096];
		while (true) {
			final int amount = inflater.inflate(buffer);
			if (amount == 0) {
				if (inflater.needsDictionary()) {
					inflater.setDictionary(dictionary);
				} else {
					break;
				}
			} else {
				bos.write(buffer, 0, amount);
			}
		}
		InputStream iis = new ByteArrayInputStream(bos.toByteArray());
		
		transformer.get().transform(new StreamSource(new InputStreamReader(iis)), domResult);
		Node documentOut = domResult.getNode();
		OMEModel model = new OMEModelImpl();
		OME ome = new OME((Element)(documentOut.getFirstChild()), model);
		model.resolveReferences();
		List<ImagePlaneDetailsStack> stacks = new ArrayList<ImagePlaneDetailsStack>();
		for (int i=0; i<ids.size(); i++) {
			final String id = ids.get(i);
			final TypedAxis [] axes = axesList.get(i);
			ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(axes);
			stack.loadFromOME(ome, id);
			stacks.add(stack);
		}
		final List<String> emptyList = Collections.emptyList();
		return new ImageSet(stacks, emptyList);
	}
	/**
	 * Create a dictionary of byte runs found in the image set data
	 * @param imageSets
	 * @return
	 * @throws IOException 
	 * @throws TransformerException 
	 * @throws TransformerFactoryConfigurationError 
	 */
	static public byte [] createCompressionDictionary(List<ImageSet> imageSets, List<String> ids) 
	throws TransformerFactoryConfigurationError, TransformerException, IOException {
		final byte [][] data = new byte [imageSets.size()][];
		Deflater deflater = new Deflater(Deflater.NO_COMPRESSION);
		for (int i=0; i<imageSets.size(); i++) {
			final ImageSet imageSet = imageSets.get(i);
			deflater.reset();
			byte [] compressed = imageSet.compress(ids, deflater);
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			InputStream is = new InflaterInputStream(new ByteArrayInputStream(compressed));
			byte [] buffer = new byte[16384];
			while (true) {
				int nBytes = is.read(buffer);
				if (nBytes <= 0) break;
				baos.write(buffer, 0, nBytes);
			}
			baos.close();
			data[i] = baos.toByteArray();
		}
		// A location within the data buffers
		class CDLoc implements Comparable<CDLoc> {
			final int imageSetIdx;
			final int dataIdx;
			final int dataLength;
			final int maxLength;
			CDLoc(int imageSetIdx, int dataIdx) {
				this.imageSetIdx = imageSetIdx;
				this.dataIdx = dataIdx;
				this.dataLength = data[imageSetIdx].length - dataIdx;
				this.maxLength = 0;
			}
			CDLoc(int imageSetIdx, int dataIdx, int dataLength, int maxLength) {
				this.imageSetIdx = imageSetIdx;
				this.dataIdx = dataIdx;
				this.dataLength = dataLength;
				this.maxLength = maxLength;
			}

			public int compareTo(CDLoc other) {
				final byte[] myData = data[imageSetIdx];
				final byte[] otherData = data[other.imageSetIdx];
				if ((imageSetIdx != other.imageSetIdx) || (dataIdx != other.dataIdx)) {
					for (int i=0; i < dataLength && i < other.dataLength - other.dataIdx; i++) {
						final int result = (int)(myData[i+dataIdx]) - (int)(otherData[i+other.dataIdx]);
						if (result != 0) return result;
					}
				}
				return dataLength - other.dataLength;
			}
			public int matchLength(CDLoc other) {
				int i;
				if ((imageSetIdx == other.imageSetIdx) && (dataIdx == other.dataIdx)) {
					return Math.min(dataLength, other.dataLength);
				}
				for (i=0; (i < dataLength) && (i < other.dataLength); i++) {
					if (data[imageSetIdx][i+dataIdx] != data[other.imageSetIdx][i+other.dataIdx]) {
						return i;
					}
				}
				return i;
			}
			public int runLength(TreeSet<CDLoc> set) {
				int result = 1;
				for (CDLoc loc:set.tailSet(this, false)) {
					if (loc.matchLength(this) < dataLength) {
						return result;
					}
					result ++;
				}
				return result;
			}
			public String toString() {
				try {
					return new String(Arrays.copyOfRange(data[imageSetIdx], dataIdx, dataIdx+dataLength));
				} catch (Exception e) {
					return super.toString();
				}
			}
		}
		TreeSet<CDLoc> set = new TreeSet<CDLoc>();
		for (int i=0; i<data.length; i++) {
			for (int j=0; j<data[i].length - 3; j++) {
				set.add(new CDLoc(i, j));
			}
		}
		TreeMap<CDLoc, Integer> runs = new TreeMap<CDLoc, Integer>();
		for (CDLoc loc:set.headSet(set.last())) {
			final CDLoc higher = set.higher(loc);
			int matchLength = loc.matchLength(higher);
			for (int i=3; i<=matchLength; i++) {
				final CDLoc key = new CDLoc(loc.imageSetIdx, loc.dataIdx, i, matchLength);
				final Entry<CDLoc, Integer> other = runs.lowerEntry(key);
				if ((other == null) || (other.getKey().matchLength(key) < i)) {
					runs.put(key, key.runLength(set));
					continue;
				}
				if (other.getKey().maxLength < key.maxLength) {
					runs.put(key, other.getValue());
				}
			}
		}
		boolean [][] toKeep = new boolean [data.length][];
		for (int i=0; i<data.length; i++) {
			toKeep[i] = new boolean[data[i].length];
		}
		for (CDLoc loc:runs.keySet()) {
			if (loc.dataLength == loc.maxLength) {
				Arrays.fill(toKeep[loc.imageSetIdx], loc.dataIdx, loc.dataIdx + loc.dataLength, true);
			}
		}
		int nBytes = 0;
		for (boolean [] a:toKeep) {
			for (boolean b:a) if (b) nBytes++;
		}
		final byte [] result = new byte[nBytes];
		int idx = 0;
		for (int i=0; i<data.length; i++) {
			byte [] d = data[i];
			boolean [] b = toKeep[i];
			for (int j=0;j<d.length; j++) {
				if (b[j]) result[idx++] = d[j];
			}
		}
		return result;
	}
	/**
	 * Convert a list of image sets into the components needed
	 * by CellProfiler's measurements
	 * 
	 * urls, pathNames, fileNames, series, index and channel are
	 * all arrays whose length is the number of columns in the
	 * image set. They are initially filled with nulls, e.g.
	 * String [][] urls = new String [imageSets.get(0).size()][];
	 * 
	 * On output, each array slot will be filled with a column of
	 * data with one entry per image set. Each of these is somewhat
	 * legacy - they are the values for the first IPD in the stack
	 * and that's usually sufficient for the typical case of either
	 * interleaved or monochrome planes.
	 * 
	 * The compressed image data can be reconstituted into an ImageSet
	 * by the decompress method.
	 * 
	 * The deflater, if present, can be cleverly primed by taking a
	 * representative image set and "compressing" it with a deflater
	 * set to use a compression level of Deflater.NO_COMPRESSION.
	 * The resulting bytes can be plugged into this deflater using
	 * setDictionary() and similarly for the inflater.
	 * 
	 * @param imageSets - the image sets that are grist for the mill
	 * @param channelNames - the names for the channels in the OME-XML
	 * @param urls - on output, the plane's URL per channel, per image set  
	 * @param pathNames - on output, the path names of each URL
	 * @param fileNames - on output, the file names of each URL
	 * @param series - on output, the series of each plane
	 * @param index - on output, the index of each plane 
	 * @param channel - on output, the channel of each plane
	 * @param dict - the data dictionary for the deflater
	 * 
	 * @return one byte array of compressed image set data per image set.
	 * @throws IOException 
	 * @throws TransformerException 
	 * @throws TransformerFactoryConfigurationError 
	 */
	public static byte [][] convertToColumns(List<ImageSet> imageSets, List<String> channelNames, 
			String [][] urls, String [][] pathNames, String [][] fileNames, int [][] series, 
			int [][] index, int [][] channel, byte [] dict) 
	throws TransformerFactoryConfigurationError, TransformerException, IOException {
		byte [][] data = new byte [imageSets.size()][];
		for (int i=0; i<channelNames.size(); i++) {
			urls[i] = new String [imageSets.size()];
			pathNames[i] = new String[imageSets.size()];
			fileNames[i] = new String[imageSets.size()];
			series[i] = new int[imageSets.size()];
			index[i] = new int[imageSets.size()];
			channel[i] = new int[imageSets.size()];
		}
		for (int i=0; i<imageSets.size(); i++) {
			final ImageSet imageSet = imageSets.get(i);
			Deflater deflater = new Deflater();
			if (dict != null) {
				deflater.setDictionary(dict);
			}
			data[i] = imageSet.compress(channelNames, deflater);
			for (int j=0; j<channelNames.size(); j++) {
				final ImagePlaneDetailsStack ipds = imageSet.get(j);
				final ImagePlane imagePlane = ipds.iterator().next().getImagePlane();
				final ImageFile imageFile = imagePlane.getImageFile();
				urls[j][i] = StringCache.intern(imageFile.getURI().toString());
				pathNames[j][i] = StringCache.intern(imageFile.getPathName());
				fileNames[j][i] = StringCache.intern(imageFile.getFileName());
				series[j][i] = imagePlane.getSeries().getSeries();
				index[j][i] = imagePlane.getIndex();
				channel[j][i] = imagePlane.getChannel();
			}
		}
		return data;
	}
	
	private static <T> List<T> icantfindafunctionthatmakesamutablelistinitializedwithasinglemember(T stack) {
		List<T> stacks = new ArrayList<T>();
		stacks.add(stack);
		return stacks;
	}
	
}
