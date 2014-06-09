/*
 * #%L
 * BSD implementations of Bio-Formats readers and writers
 * %%
 * Copyright (C) 2005 - 2014 Open Microscopy Environment:
 *   - Board of Regents of the University of Wisconsin-Madison
 *   - Glencoe Software, Inc.
 *   - University of Dundee
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package loci.formats.in;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import ome.xml.model.primitives.PositiveInteger;

import loci.common.DataTools;
import loci.common.RandomAccessInputStream;
import loci.formats.CoreMetadata;
import loci.formats.FormatException;
import loci.formats.FormatReader;
import loci.formats.FormatTools;
import loci.formats.MetadataTools;
import loci.formats.meta.MetadataStore;
import loci.formats.tiff.IFD;
import loci.formats.tiff.TiffParser;

/**
 * @author Lee Kamentsky
 * 
 * Reader for the Amris ImageStream / FlowSight file format
 * 
 * The cif file format is tiff-like, but doesn't adhere to
 * the TIFF standard, so we use the TiffParser where we can, 
 * but do not use the TiffReader hierarchy.
 *
 */
public class FlowSightReader extends FormatReader {
	final private static int CHANNEL_COUNT_TAG = 33000;
	final private static int ACQUISITION_TIME_TAG = 33004;
	final private static int CHANNEL_NAMES_TAG = 33007;
	final private static int CHANNEL_DESCS_TAG = 33008;
	final private static int METADATA_XML_TAG = 33027;
	
	final private static int GREYSCALE_COMPRESSION = 30817;
	final private static int BITMASK_COMPRESSION = 30818;
	
	/**
	 * The tags that must be present for this reader
	 * to function properly
	 */
	final private static int [] MINIMAL_TAGS = {
		CHANNEL_COUNT_TAG,
		CHANNEL_NAMES_TAG,
		CHANNEL_DESCS_TAG
	};
	
	/**
	 * This stream is opened on the file supplied
	 * via "setId".
	 */
	private RandomAccessInputStream in;
	
	private TiffParser tiffParser;
	private long [] ifdOffsets;
	
	private String [] channelNames;
	private String [] channelDescs;
	
	public FlowSightReader() {
		super("FlowSight format", "cif");
		saveOriginalMetadata = false;
	}

	/* (non-Javadoc)
	 * @see loci.formats.FormatReader#isThisType(loci.common.RandomAccessInputStream)
	 */
	@Override
	public boolean isThisType(RandomAccessInputStream stream)
			throws IOException {
		TiffParser tiffParser = new TiffParser(stream);
		if (! tiffParser.isValidHeader()) return false;
		IFD ifd = tiffParser.getFirstIFD();
		if (ifd == null) return false;
		tiffParser.fillInIFD(ifd);
		for (int tag: MINIMAL_TAGS)
			try {
				if (ifd.getIFDStringValue(tag) == null) return false;
			} catch (FormatException e) {
				return false;
			}
		return true;
	}

	/* (non-Javadoc)
	 * @see loci.formats.FormatReader#initFile(java.lang.String)
	 */
	@Override
	protected void initFile(String id) throws FormatException, IOException {
		super.initFile(id);
	    in = new RandomAccessInputStream(id);
	    tiffParser = new TiffParser(in);
	    tiffParser.setDoCaching(false);
	    tiffParser.setUse64BitOffsets(false);
	    final Boolean littleEndian = tiffParser.checkHeader();
	    if (littleEndian == null) {
	      throw new FormatException("Invalid FlowSight file");
	    }
	    final boolean little = littleEndian.booleanValue();
	    in.order(little);
	    
	    LOGGER.info("Reading IFDs");

	    ifdOffsets = tiffParser.getIFDOffsets();

	    if (ifdOffsets.length < 2) {
	      throw new FormatException("No IFDs found");
	    }
	    
	    LOGGER.info("Populating metadata");

	    /*
	     * The first IFD contains file-scope metadata
	     */
	    final IFD ifd0 = tiffParser.getFirstIFD();
	    tiffParser.fillInIFD(ifd0);
	    final int channelCount = ifd0.getIFDIntValue(CHANNEL_COUNT_TAG);
	    final String channelNamesString = ifd0.getIFDStringValue(CHANNEL_NAMES_TAG);
	    channelNames = channelNamesString.split("\\|");
	    if (channelNames.length != channelCount) {
	    	throw new FormatException(String.format(
	    			"Channel count (%d) does not match number of channel names (%d) in string \"%s\"",
	    			channelCount, channelNames.length, channelNamesString));
	    }
	    LOGGER.debug(String.format(
	    		"Found %d channels: %s", 
	    		channelCount, channelNamesString.replace('|', ',')));
	    final String channelDescsString = ifd0.getIFDStringValue(CHANNEL_DESCS_TAG);
	    channelDescs =  channelDescsString.split("\\|");
	    if (channelDescs.length != channelCount) {
	    	throw new FormatException(String.format(
	    			"Channel count (%d) does not match number of channel descriptions (%d) in string \"%s\"",
	    			channelCount, channelDescs.length, channelDescsString));
	    }
	    /*
	     * Scan the remaining IFDs
	     * 
	     * Unfortunately, each image can have a different width and height
	     * and the images and masks have a different bit depth, so in the
	     * OME scheme of things, we get one series per plane.
	     */
	    for (int idxOff=1; idxOff<ifdOffsets.length;idxOff++) {
	    	// TODO: Record the channel names
	    	final long offset = ifdOffsets[idxOff];
	    	final boolean first=(idxOff == 1);
	    	final IFD ifd = tiffParser.getIFD(offset);
	    	tiffParser.fillInIFD(ifd);
	        CoreMetadata ms = first?core.get(0):new CoreMetadata();
	        ms.rgb = false;
	        ms.interleaved = false;
	        ms.littleEndian = ifd0.isLittleEndian();
	        ms.sizeX = (int) ifd.getImageWidth() / channelCount;
	        ms.sizeY = (int) ifd.getImageLength();
	        ms.sizeZ = 1;
	        ms.sizeC = channelCount;
	        ms.sizeT = 1;
	        ms.indexed = false;
	        ms.dimensionOrder = "XYCZT";
	        ms.bitsPerPixel = ifd.getIFDIntValue(IFD.BITS_PER_SAMPLE);
	        ms.pixelType = (ms.bitsPerPixel == 8)?FormatTools.UINT8:FormatTools.UINT16;
	        ms.imageCount = channelCount;
	        ms.resolutionCount = 1;
	        ms.thumbnail = false;
	        ms.metadataComplete = true;
	        if (! first) {
	        	core.add(ms);
	        }
	    }
	    /*
	     * Run through the metadata store, setting the channel names
	     * for all the series.
	     */
	    final MetadataStore store = getMetadataStore();
	    String [] maskDescs = new String [channelCount];
	    for (int i=0; i<channelCount; i++) {
	    	maskDescs[i] = channelDescs[i] + "Mask";
	    }
	    MetadataTools.populatePixels(store, this);
	    for (int series=0; series < ifdOffsets.length-1; series++) {
    		final boolean isMask = (core.get(series).pixelType == FormatTools.UINT8);
    		String [] descs = isMask?maskDescs:channelDescs;
	    	for (int channel=0; channel < channelCount; channel++) {
	    		store.setChannelName(descs[channel], series, channel);
	    		store.setChannelID(channelNames[channel], series, channel);
	    	}
	    }
	    
	}

	@Override
	public void close(boolean fileOnly) throws IOException {
		super.close(fileOnly);
		tiffParser = null;
		in = null;
		ifdOffsets = null;
		channelNames = null;
		channelDescs = null;
	}

	@Override
	public byte[] openBytes(int no, byte[] buf, int x, int y, int w, int h)
			throws FormatException, IOException {
		if (no > getChannelCount()) {
			throw new FormatException("Only one plane per series");
		}
		final int idx = getSeries() + 1;
		final IFD ifd = tiffParser.getIFD(ifdOffsets[idx]);
		final int imageWidth = (int)(ifd.getImageWidth());
		final int imageHeight = (int)(ifd.getImageLength());
		final int wOff = x + no * imageWidth / getChannelCount();
		if ((y+h > imageHeight) || (x+w > imageWidth / getChannelCount()) ) {
			throw new FormatException("Requested tile dimensions extend beyond those of the image.");
		}
		final int compression = ifd.getIFDIntValue(IFD.COMPRESSION);
		byte [] tempBuffer; /* NB - these images are very small */
		switch(compression) {
		case GREYSCALE_COMPRESSION:
			tempBuffer = openGreyscaleBytes(ifd, imageWidth, imageHeight);
			break;
		case BITMASK_COMPRESSION:
			tempBuffer = openBitmaskBytes(ifd, imageWidth, imageHeight);
			break;
		default:
			throw new FormatException(String.format("Unknown compression code: %d", compression));
		}
		final int bytesPerSample = ifd.getIFDIntValue(IFD.BITS_PER_SAMPLE) / 8;
		for (int yy=y; yy<y+h; yy++) {
			final int srcOff = bytesPerSample * (wOff + yy * imageWidth);
			final int destOff = bytesPerSample * (yy-y) * w;
			System.arraycopy(tempBuffer, srcOff, buf, destOff, w * bytesPerSample);
		}
		return buf;
	}

	/**
	 * Decode the whole IFD plane using bitmask compression
	 * 
	 * @param ifd - the IFD to decode
	 * @param imageWidth the width of the IFD plane
	 * @param imageHeight the height of the IFD plane
	 * @return a byte array of length imageWidth * imageHeight
	 *         containing the uncompressed data
	 * @throws FormatException 
	 */
	private byte[] openBitmaskBytes(IFD ifd, int imageWidth, int imageHeight) throws FormatException {
		final byte [] uncompressed = new byte[imageWidth * imageHeight];
		final long [] stripByteCounts = ifd.getIFDLongArray(IFD.STRIP_BYTE_COUNTS);
		final long [] stripOffsets = ifd.getIFDLongArray(IFD.STRIP_OFFSETS);
		int off = 0;
		for (int i=0; i<stripByteCounts.length; i++) {
			try {
				in.seek(stripOffsets[i]);
				for (int j=0; j<stripByteCounts[i]; j+=2) {
					byte value = in.readByte();
					int runLength = (in.readByte() & 0xFF)+1;
					if (off + runLength > uncompressed.length) {
						throw new FormatException("Unexpected buffer overrun encountered when decompressing bitmask data");
					}
					Arrays.fill(uncompressed, off, off+runLength, value);
					off += runLength;
				}
			} catch (IOException e) {
				LOGGER.error("Caught exception while reading bitmask IFD data", e);
				throw new FormatException(String.format("Error in FlowSight file format: %s", e.getMessage()));
			}
		}
		if (off != uncompressed.length) throw new FormatException("Buffer shortfall encountered when decompressing bitmask data");
		return uncompressed;
	}
	
	/**
	 * Decode the whole IFD plane using greyscale compression
	 * 
	 * @param ifd
	 * @param imageWidth
	 * @param imageHeight
	 * @return
	 * @throws FormatException 
	 */
	private byte[] openGreyscaleBytes(final IFD ifd, final int imageWidth, final int imageHeight) throws FormatException {
		final FormatException [] formatException = new FormatException[1];
		final long [] stripByteCounts = ifd.getIFDLongArray(IFD.STRIP_BYTE_COUNTS);
		final long [] stripOffsets = ifd.getIFDLongArray(IFD.STRIP_OFFSETS);
		Iterator<Short> diffs = new Iterator<Short> () {
			int index = -1;
			int offset = 0;
			int count = 0;
			byte currentByte;
			int nibbleIdx = 2;
			short value = 0;
			short shift = 0;
			boolean bHasNext = (formatException[0] != null);
			boolean loaded = bHasNext;
			
			@Override
			public boolean hasNext() {
				if (loaded) return bHasNext;
				shift = 0;
				value = 0;
				while (! loaded) {
					byte nibble;
					try {
						nibble = getNextNibble();
						value += ((short) (nibble & 0x7) ) << shift;
						shift += 3;
						if ((nibble & 0x8) == 0) {
							loaded = true;
							bHasNext = true;
							if ((nibble & 0x4) != 0) {
								/*
								 * The number is negative
								 * and the bits at 1 << shift and above
								 * should all be "1". This does it.
								 */
								value |= - (1 << shift);
							}
						}
					} catch (IOException e) {
						LOGGER.error("IOException during read of greyscale image", e);
						formatException[0] = new FormatException(
								String.format("Error in FlowSight format: %s", e.getMessage()));
						loaded = true;
						bHasNext = false;
					} catch (FormatException e) {
						LOGGER.error("Format exception during read of greyscale image", e);
						formatException[0] = e;
						loaded = true;
						bHasNext = false;
					}
				}
				return bHasNext;
			}
			
			private byte getNextNibble() throws IOException, FormatException {
				if (nibbleIdx >= 2) {
					if (! getNextByte()) {
						return (byte)0xff;
					}
					nibbleIdx = 0;
				}
				if (nibbleIdx++ == 0) {
					return (byte)(currentByte & 0x0f);
				} else {
					return (byte)(currentByte >> 4);
				}
			}
	
			private boolean getNextByte() throws IOException, FormatException {
				while (offset == count) {
					index++;
					if (index == stripByteCounts.length) {
						loaded = true;
						bHasNext = false;
						return false;
					}
					in.seek(stripOffsets[index]);
					offset = 0;
					count = (int)stripByteCounts[index];
				}
				currentByte = in.readByte();
				offset++;
				return true;
			}
	
			@Override
			public Short next() {
				if (! hasNext()) throw new IndexOutOfBoundsException("Tried to read past end of IFD data");
				loaded = false;
				return value;
			}
	
			@Override
			public void remove() {
				throw new UnsupportedOperationException();
			}
						
		};
		byte [] buffer = new byte [imageWidth * imageHeight * 2];
		short [] lastRow = new short[imageWidth];
		short [] thisRow = new short[imageWidth];
		int index = 0;
		for (int y=0; y<imageHeight; y++) {
			for (int x = 0; x<imageWidth; x++) {
				if (x != 0) {
					thisRow[x] = (short)(diffs.next() + lastRow[x] + thisRow[x-1] - lastRow[x-1]);  
				} else {
					thisRow[x] = (short)(diffs.next() + lastRow[x]);
				}
				DataTools.unpackBytes(thisRow[x], buffer, index, 2, in.isLittleEndian());
				index += 2;
			}
			final short [] temp = lastRow;
			lastRow = thisRow;
			thisRow = temp;
		}
		return buffer;
	}

	/**
	 * @return number of channels per series
	 */
	private int getChannelCount() {
		return channelNames.length;
	}

}
