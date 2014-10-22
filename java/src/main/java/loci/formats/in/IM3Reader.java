/*#%L
 * OME Bio-Formats package for reading and converting biological file formats.
 * %%
 * Copyright (C) 2005 - 2013 Open Microscopy Environment:
 *   - Board of Regents of the University of Wisconsin-Madison
 *   - Glencoe Software, Inc.
 *   - University of Dundee
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 2 of the 
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public 
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-2.0.html>.
 * #L%
 */

package loci.formats.in;

import java.io.IOException;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ome.xml.model.enums.DimensionOrder;
import loci.common.IRandomAccess;
import loci.common.Location;
import loci.common.RandomAccessInputStream;
import loci.formats.CoreMetadata;
import loci.formats.FormatException;
import loci.formats.FormatReader;
import loci.formats.FormatTools;
import loci.formats.MetadataTools;

/**
 * @author Lee Kamentsky
 * 
 * This is a FormatReader for the Perkin-Elmer Nuance
 * line of multispectral imaging microscopes' .im3
 * file format. The .im3 format stores 63 planes
 * representing intensity measurements taken on
 * separate spectral bands. The pixel intensity at
 * a single band might be used as a proxy for
 * fluorophore presence, but the best results are
 * obtained if the whole spectrum is used to
 * synthesize an intensity. The most efficient
 * strategy is to use IM3Reader.openRaw() to
 * fetch the entire image which may then be scanned
 * sequentially to create one planar image
 * per desired signal.
 * 
 * IM3Reader may be run as a Java application to
 * dump a per-record description of the file to
 * standard output
 *
 */
public class IM3Reader extends FormatReader {
	/**
	 *  Logger for outputting summary diagnostics.
	 */
	private static final Logger LOGGER = LoggerFactory.getLogger(IM3Reader.class);
	/**
	 * First 4 bytes of file is "1985"
	 */
	private static final int COOKIE = 1985;
	/**
	 * A record that is a container of
	 * other records.
	 */
	private static final int REC_CONTAINER = 0;
	/**
	 * Image format
	 * int: pixel type? 3 = RGBA
	 * int: width
	 * int: height
	 * int: depth
	 * data bytes follow
	 */
	@SuppressWarnings("unused")
	private static final int REC_IMAGE = 1;
	/**
	 * A nested record of records 
	 */
	@SuppressWarnings("unused")
	private static final int REC_NESTED = 3;
	/**
	 * 32-bit integer values 
	 */
	private static final int REC_INT = 6;
	/*
	 * Floating point values
	 */
	private static final int REC_FLOAT = 7;
	private static final int REC_BOOLEAN = 9;
	/**
	 * A string
	 * int: ?
	 * int: length
	 * 8-byte characters follow
	 */
	private static final int REC_STRING=10;
	/*
	 * Container fields.
	 */
	private static final String FIELD_DATA_SET="DataSet";
	@SuppressWarnings("unused")
	private static final String FIELD_TIMESTAMP="TimeStamp";
	@SuppressWarnings("unused")
	private static final String FIELD_AUX_FLAGS="AuxFlags";
	@SuppressWarnings("unused")
	private static final String FIELD_NUANCE_FLAGS="NuanceFlags";
	private static final String FIELD_SPECTRA="Spectra";
	private static final String FIELD_VALUES="Values";
	@SuppressWarnings("unused")
	private static final String FIELD_PROTOCOL="Protocol";
	@SuppressWarnings("unused")
	private static final String FIELD_OBJECTIVE="Objective";
	@SuppressWarnings("unused")
	private static final String FIELD_SPECTRAL_BASIS_INFO="SpectralBasisInfo";
	@SuppressWarnings("unused")
	private static final String FIELD_FILTER_PAIR="FilterPair";
	@SuppressWarnings("unused")
	private static final String FIELD_FIXED_FILTER="FixedFilter";
	@SuppressWarnings("unused")
	private static final String FIELD_BANDS="Bands";
	private static final String FIELD_SPECTRAL_LIBRARY="SpectralLibrary";
	private static final String FIELD_SPECTRUM="Spectrum";
	/*
	 * Image fields
	 */
	@SuppressWarnings("unused")
	private static final String FIELD_THUMBNAIL="Thumbnail";
	private static final String FIELD_DATA="Data";
	/*
	 * Int fields
	 */
	@SuppressWarnings("unused")
	private static final String FIELD_FILE_VERSION="FileVersion";
	@SuppressWarnings("unused")
	private static final String FIELD_CLASS_ID="ClassID";
	@SuppressWarnings("unused")
	private static final String FIELD_TYPE_ID="TypeID";
	private static final String FIELD_SHAPE="Shape";
	@SuppressWarnings("unused")
	private static final String FIELD_BAND_INDEX="BandIndex";
	/*
	 * String fields
	 */
	private static final String FIELD_NAME="Name";
	@SuppressWarnings("unused")
	private static final String FIELD_SAMPLE_ID="SampleID";
	@SuppressWarnings("unused")
	private static final String FIELD_USER_COMMENTS="UserComments";
	@SuppressWarnings("unused")
	private static final String FIELD_SOURCE_FILE_NAME="SourceFileName";
	@SuppressWarnings("unused")
	private static final String FIELD_PROXY_PARENT_FILE_NAME="ProxyParentFileName";
	@SuppressWarnings("unused")
	private static final String FIELD_MANUFACTURER="Manufacturer";
	@SuppressWarnings("unused")
	private static final String FIELD_PART_NUMBER="PartNumber";
	/*
	 * Float fields
	 */
	@SuppressWarnings("unused")
	private static final String FIELD_MILLIMETERS_PER_PIXEL="MillimetersPerPixel";
	@SuppressWarnings("unused")
	private static final String FIELD_EXPOSURE="Exposure";
	@SuppressWarnings("unused")
	private static final String FIELD_WAVELENGTH="Wavelength";
	private static final String FIELD_WAVELENGTHS="Wavelengths";
	private static final String FIELD_MAGNITUDES="Magnitudes";
	/*
	 * Misc fields
	 */
	@SuppressWarnings("unused")
	private static final String FIELD_HOMOGENEOUS="Homogeneous";
	/*
	 * Records for the current file
	 */
	private List<IM3Record> records;
	/*
	 * Data sets for the current file
	 */
	private List<ContainerRecord> dataSets;
	/*
	 * Spectrum records for a spectral library
	 */
	private List<Spectrum> spectra;
	
	/*
	 * The data from the current series' file.
	 */
	private byte [] data;
	
	/**
	 * Construct an uninitialized reader of .im3 files.
	 */
	public IM3Reader() {
		super("Perkin-Elmer Nuance IM3", "im3");
	}

	@Override
	public byte[] openBytes(int no, byte[] buf, int x, int y, int w, int h)
			throws FormatException, IOException {
		FormatTools.checkPlaneParameters(this, no, buf.length, x, y, w, h);
		if (data == null) {
			data = openRaw();
		}
		if (data == null) return null;
		
		final int srcWidth = getSizeX();
		final int srcChannels = getSizeC();
		int idx = 0;
		int offset = ((x + y * srcWidth) * srcChannels + no) * 2;
		for (int hidx=0; hidx < h; hidx++) {
			int roffset = offset + hidx * srcWidth * srcChannels * 2;
			for (int widx=0; widx < w; widx++) {
				buf[idx++] = data[roffset];
				buf[idx++] = data[roffset+1];
				roffset += srcChannels * 2;
			}
		}
		return buf;
	}
	
	/**
	 * Open the current series in raw-mode, returning the
	 * interleaved image bytes. The data format for a pixel is
	 * a run of 63 unsigned short little endian values.
	 * 
	 * @return a byte array containing the data organized by
	 *         spectral channel, then x, then y. Returns null
	 *         if, for some incomprehensible reason, the DATA
	 *         block was missing.
	 * @throws IOException 
	 */
	public byte [] openRaw() throws IOException {
		IRandomAccess is = Location.getHandle(getCurrentFile(), false);
		is.setOrder(ByteOrder.LITTLE_ENDIAN);
		final ContainerRecord dataSet = dataSets.get(getSeries());
		for (IM3Record subRec:dataSet.parseChunks(is)){
			if (subRec.name.equals(FIELD_DATA)) {
				is.seek(subRec.offset+4);
				int width = is.readInt();
				int height = is.readInt();
				int channels = is.readInt();
				final byte [] result = new byte [width * height * channels * 2];
				is.read(result);
				return result;
			}
		}
		return null;
	}

	/**
	 * If a Nuance file is a spectral library set (.csl) file,
	 * there is a spectral library inside that contains a profile
	 * of spectral bin magnitudes for each of several measured
	 * fluorophores (or auto fluorescence). This method finds
	 * the spectral library and returns the spectra inside.
	 * 
	 * @return a list of the Spectrum records contained in the library
	 */
	public List<Spectrum> getSpectra() {
		return spectra;
	}
	/* (non-Javadoc)
	 * @see loci.formats.FormatReader#initFile(java.lang.String)
	 */
	@Override
	protected void initFile(String id) throws FormatException, IOException {
		super.initFile(id);
		IRandomAccess is = Location.getHandle(id, false);
		is.setOrder(ByteOrder.LITTLE_ENDIAN);
		final int cookie = is.readInt();
		if (cookie != COOKIE) {
			throw new FormatException(String.format("Expected file cookie of %d, but got %d.", COOKIE, cookie));
		}
		long fileLength = is.length();
		records = new ArrayList<IM3Record>();
		dataSets = new ArrayList<ContainerRecord>();
		spectra = new ArrayList<Spectrum>();
		core = new ArrayList<CoreMetadata>();

		while (is.getFilePointer() < fileLength) {
			final IM3Record rec = parseRecord(is);
			if (rec == null) {
				if (is.getFilePointer() > fileLength-16) break;
				/*
				 * # of bytes in chunk.
				 */
				@SuppressWarnings("unused")
				final int chunkLength = is.readInt();
				/*
				 * Is always zero? Chunk #?
				 */
				@SuppressWarnings("unused")
				final int unknown = is.readInt();
				/*
				 * Is always one? Chunk #?
				 */
				@SuppressWarnings("unused")
				final int unknown1 = is.readInt();
				/*
				 * # of records to follow
				 */
				@SuppressWarnings("unused")
				final int nRecords = is.readInt();
			} else {
				if (rec instanceof ContainerRecord) {
					final ContainerRecord bRec = (ContainerRecord)rec;
					for (IM3Record subDS:bRec.parseChunks(is)) {
						if ((subDS instanceof ContainerRecord) && (subDS.name.equals(FIELD_DATA_SET))) {
							final ContainerRecord bSubDS = (ContainerRecord)subDS;
							for (IM3Record subSubDS:bSubDS.parseChunks(is)) {
								if (subSubDS instanceof ContainerRecord) {
									final ContainerRecord bDataSet = (ContainerRecord) subSubDS;
									dataSets.add(bDataSet);
									List<IM3Record> subRecs = bDataSet.parseChunks(is);
									final CoreMetadata cm = new CoreMetadata();
									cm.dimensionOrder = DimensionOrder.XYCZT.getValue();
									cm.littleEndian = true;
									// TODO: Detect pixel type
									cm.pixelType = FormatTools.UINT16;
									for (IM3Record subRec:subRecs){
										if (subRec.name.equals(FIELD_SHAPE) && (subRec instanceof IntIM3Record)) {
											final IntIM3Record iRec = (IntIM3Record)subRec;
											cm.sizeX = iRec.getEntry(is, 0);
											cm.sizeY = iRec.getEntry(is, 1);
											cm.sizeC = iRec.getEntry(is, 2);
											cm.sizeZ = 1;
											cm.sizeT = 1;
											cm.imageCount = cm.sizeC;
											cm.metadataComplete = true;
										}
									}
									core.add(cm);
								}
							}
						} else if ((subDS instanceof ContainerRecord) && subDS.name.equals(FIELD_SPECTRAL_LIBRARY)) {
							/*
							 * SpectralLibrary
							 *   (unnamed container record)
							 *       Spectra
							 *           Keys (integers)
							 *           Values
							 *               (unnamed container record for spectrum #1)
							 *               (unnamed container record for spectrum #2)...
							 *                
							 */
							for (IM3Record slContainer:((ContainerRecord)subDS).parseChunks(is)) { /* unnamed container */
								if (slContainer instanceof ContainerRecord) {
									for (IM3Record slSpectra:((ContainerRecord)slContainer).parseChunks(is)) {
										if ((slSpectra instanceof ContainerRecord) && (slSpectra.name.equals(FIELD_SPECTRA))) {
											for (IM3Record slRec:((ContainerRecord)slSpectra).parseChunks(is)) {
												if (slRec.name.equals(FIELD_VALUES) && (slRec instanceof ContainerRecord)) {
													for (IM3Record spectrumRec:((ContainerRecord)slRec).parseChunks(is)) {
														if (spectrumRec instanceof ContainerRecord) {
															spectra.add(new Spectrum(is, (ContainerRecord)spectrumRec));
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
				records.add(rec);
			}
		}
		MetadataTools.populatePixels(getMetadataStore(), this);
	}
	static private final String EMPTY_STRING = new String();
	/**
	 * Parse a string from the IM3 file at the current file pointer loc
	 * 
	 * @param is stream to read from
	 * @return parsed string or null for string of zero length
	 * @throws IOException
	 */
	static protected String parseString(IRandomAccess is) throws IOException {
		final int nameLength = is.readInt();
		if (nameLength == 0) return EMPTY_STRING;
		final byte [] buf = new byte [nameLength];
		is.read(buf);
		return new String(buf, loci.common.Constants.ENCODING);
	}
	
	/**
	 * Parse an IM3 record at the current file pointer location
	 * 
	 * @param is random access stream, pointing at the record's start
	 *        (the length-quadword of the record's tag name)
	 * @return an IM3Record or subclass depending on the record's type
	 * @throws IOException on file misparsing leading to overrun and other
	 */
	private IM3Record parseRecord(IRandomAccess is) throws IOException {
		final String name = parseString(is);
		if (name == null) return null;
		final int recLength = is.readInt()-8;
		final int recType = is.readInt();
		final long offset = is.getFilePointer();
		is.skipBytes(recLength);
		switch(recType) {
			case REC_CONTAINER:
				return new ContainerRecord(name, recType, offset, recLength);
			case REC_STRING:
				return new StringIM3Record(name, recType, offset, recLength);
			case REC_INT:
				return new IntIM3Record(name, recType, offset, recLength);
			case REC_FLOAT:
				return new FloatIM3Record(name, recType, offset, recLength);
			case REC_BOOLEAN:
				return new BooleanIM3Record(name, recType, offset, recLength);
		}
		return new IM3Record(name, recType, offset, recLength);
	}
	/* (non-Javadoc)
	 * @see loci.formats.FormatReader#isThisType(loci.common.RandomAccessInputStream)
	 */
	@Override
	public boolean isThisType(RandomAccessInputStream stream)
			throws IOException {
		stream.seek(0);
		return (stream.readInt() == COOKIE);
	}
	protected class IM3Record {
		final String name;
		final int type;
		final long offset;
		final int length;
		IM3Record(String name, int type, long offset, int length) {
			this.name = name;
			this.type = type;
			this.offset=offset;
			this.length=length;
		}
		/**
		 * Write a summary of the contents of the record
		 * 
		 * @param is
		 * @throws IOException 
		 */
		public void writeSummary(IRandomAccess is, String indentation) throws IOException {
			is.seek(offset);
			LOGGER.info(indentation + toString());
			for (int i=0; (i<length) && (i < 256); i+= 32) {
				StringBuilder msg = new StringBuilder(indentation + String.format("%02x:", i));
				for (int j=i;(j < length) &&(j < i+32); j++) {
					msg.append(String.format(" %02x", is.readByte()));
				}
				LOGGER.info(msg.toString());
			}
		}
		/* (non-Javadoc)
		 * @see java.lang.Object#toString()
		 */
		@Override
		public String toString() {
			return String.format("[%s: type=%d, offset=%d, length=%d]", name, type, offset, length);
		}
		
	}
	/**
	 * @author Lee Kamentsky
	 *
	 * A Container3Record is a nesting container for
	 * other records. In the IM3 format, records are often grouped
	 * under a ContainerRecord with a blank tagname.
	 */
	protected class ContainerRecord extends IM3Record {

		ContainerRecord(String name, int type, long offset, int length) {
			super(name, type, offset, length);
		}
		/**
		 * Parse and return the sub-records for the record container
		 * 
		 * @param is
		 * @return
		 * @throws IOException
		 */
		List<IM3Record> parseChunks(IRandomAccess is) throws IOException {
			long oldOffset = is.getFilePointer();
			is.seek(offset+8);
			long end = offset+length;
			List<IM3Record> recs = new ArrayList<IM3Record>();
			while(is.getFilePointer() < end-8) {
				final IM3Record rec = parseRecord(is);
				if (rec != null)
					recs.add(rec);
			}
			is.seek(oldOffset);
			return recs;
		}
		
		/* (non-Javadoc)
		 * @see loci.formats.in.IM3Reader.IM3Record#writeSummary(loci.common.IRandomAccess, java.lang.String)
		 */
		public void writeSummary(IRandomAccess is, String indentation) throws IOException {
			is.seek(offset);
			LOGGER.info(indentation + toString());
			for (IM3Record rec:parseChunks(is)) {
				rec.writeSummary(is, indentation + "  ");
			}
		}
	}
	/**
	 * @author Lee Kamentsky
	 *
	 * An integer array record
	 */
	protected class IntIM3Record extends IM3Record {

		public IntIM3Record(String name, int recType, long offset, int recLength) {
			super(name, recType, offset, recLength);
		}
		
		/**
		 * Get the # of integer values in this record
		 * @param is 
		 * 
		 * @return number of integer values contained in the record
		 * @throws IOException 
		 */
		public int getNumEntries(IRandomAccess is) throws IOException {
			long oldPos = is.getFilePointer();
			try {
				is.seek(offset);
				final int code = is.readInt();
				if (code == 0) return 1;
				return is.readInt();
			} finally {
				is.seek(oldPos);
			}
		}
		
		/**
		 * Get the integer value at the given index
		 * 
		 * @param is the stream for the IM3 file
		 * @param index the zero-based index of the entry to retrieve
		 * @return the value stored in the indexed slot of the record
		 * @throws IOException
		 */
		public int getEntry(IRandomAccess is, int index) throws IOException {
			long oldPos = is.getFilePointer();
			try {
				is.seek(offset);
				if (is.readInt() == 0) return is.readInt();
				is.seek(offset+index*4+8);
				return is.readInt();
			} finally {
				is.seek(oldPos);
			}
		}
		/* (non-Javadoc)
		 * @see loci.formats.in.IM3Reader.IM3Record#writeSummary(loci.common.IRandomAccess, java.lang.String)
		 */
		public void writeSummary(IRandomAccess is, String indentation) throws IOException {
			is.seek(offset);
			LOGGER.info(indentation + toString());
			final int length = getNumEntries(is);
			for (int i=0; (i < length) && (i < 256); i+=16) {
				StringBuilder msg = new StringBuilder(indentation + String.format("%02x:", i));
				for (int j=i; (j<i+16) && (j<length); j++) {
					msg.append(String.format(" %7d", getEntry(is, j)));
				}
				LOGGER.info(msg.toString());
			}
		}
	}
	/**
	 * @author Lee Kamentsky
	 *
	 * A 4-byte floating point array record
	 * 
	 * The record format is
	 * int32 - unknown
	 * int32 - # of floats
	 * float32s - values
	 */
	protected class FloatIM3Record extends IM3Record {

		public FloatIM3Record(String name, int recType, long offset, int recLength) {
			super(name, recType, offset, recLength);
		}
		
		/**
		 * Get the # of floating-point values in this record
		 * 
		 * @return number of integer values contained in the record
		 * @throws IOException 
		 */
		public int getNumEntries(IRandomAccess is) throws IOException {
			long oldPos = is.getFilePointer();
			try {
				is.seek(offset);
				if (is.readInt() == 0) return 1;
				return is.readInt();
			} finally {
				is.seek(oldPos);
			}
		}
		
		/**
		 * Get the floating-point value at the given index
		 * 
		 * @param is the stream for the IM3 file
		 * @param index the zero-based index of the entry to retrieve
		 * @return the value stored in the indexed slot of the record
		 * @throws IOException
		 */
		public float getEntry(IRandomAccess is, int index) throws IOException {
			long oldPos = is.getFilePointer();
			try {
				is.seek(offset);
				if (is.readInt() == 0) return is.readFloat();
				is.seek(offset+8+index*4);
				return is.readFloat();
			} finally {
				is.seek(oldPos);
			}
		}
		/**
		 * Return all entries as an array 
		 * @param is handle to file
		 * @return an array of the stored values
		 * @throws IOException 
		 */
		public float [] getEntries(IRandomAccess is) throws IOException {
			final long oldPos = is.getFilePointer();
			try {
				float [] values = new float[getNumEntries(is)];
				is.seek(offset+8);
				for (int index=0; index < values.length; index++) {
					values[index] = is.readFloat();
				}
				return values;
			} finally {
				is.seek(oldPos);
			}
			
		}
		/* (non-Javadoc)
		 * @see loci.formats.in.IM3Reader.IM3Record#writeSummary(loci.common.IRandomAccess, java.lang.String)
		 */
		public void writeSummary(IRandomAccess is, String indentation) throws IOException {
			is.seek(offset);
			LOGGER.info(indentation + toString());
			final int length = getNumEntries(is);
			for (int i=0; (i < length) && (i < 256); i+=16) {
				StringBuilder msg = new StringBuilder(indentation + String.format("%02x:", i));
				for (int j=i; (j<i+16) && (j<length); j++) {
					msg.append(String.format(" %4.4f", getEntry(is, j)));
				}
				LOGGER.info(msg.toString());
			}
		}
	}
	/**
	 * A record containing boolean values
	 * 
	 * @author Lee Kamentsky
	 *
	 */
	protected class BooleanIM3Record extends IM3Record {
		public BooleanIM3Record(String name, int recType, long offset, int recLength) {
			super(name, recType, offset, recLength);
		}
		
		/**
		 * Get the # of boolean values in this record
		 * 
		 * @return number of boolean values contained in the record
		 * @throws IOException 
		 */
		public int getNumEntries(IRandomAccess is) throws IOException {
			long oldPos = is.getFilePointer();
			try {
				is.seek(offset+4);
				return is.readInt();
			} finally {
				is.seek(oldPos);
			}
		}
		
		/**
		 * Get the boolean value at the given index
		 * 
		 * @param is the stream for the IM3 file
		 * @param index the zero-based index of the entry to retrieve
		 * @return the value stored in the indexed slot of the record
		 * @throws IOException
		 */
		public boolean getEntry(IRandomAccess is, int index) throws IOException {
			long oldPos = is.getFilePointer();
			try {
				is.seek(offset+8+index);
				return (is.readByte() != 0);
			} finally {
				is.seek(oldPos);
			}
		}
		
	}
	/**
	 * @author Lee Kamentsky
	 *
	 * A record whose value is a string.
	 */
	protected class StringIM3Record extends IM3Record {
		public StringIM3Record(String name, int recType, long offset, int recLength) {
			super(name, recType, offset, recLength);
		}
		
		/**
		 * Return the string value for this record
		 * 
		 * @param is an open handle on the .IM3 file
		 * @return the string value stored in the record
		 * @throws IOException
		 */
		public String getValue(IRandomAccess is) throws IOException {
			final long oldPos = is.getFilePointer();
			try {
				is.seek(offset+4);
				return parseString(is);
			} finally {
				is.seek(oldPos);
			}
		}

		/* (non-Javadoc)
		 * @see loci.formats.in.IM3Reader.IM3Record#writeSummary(loci.common.IRandomAccess, java.lang.String)
		 */
		@Override
		public void writeSummary(IRandomAccess is, String indentation) throws IOException {
			LOGGER.info(indentation + toString());
			LOGGER.info(indentation + String.format("Value = %s", getValue(is)));
		}
		
	}
	/**
	 * @author Lee Kamentsky
	 * 
	 * Represents a Spectrum record within a SpectralLibrary
	 *
	 */
	static public class Spectrum {
		private String name;
		private float [] wavelengths;
		private float [] magnitudes;
		/**
		 * Construct a spectrum by parsing a Spectrum record
		 * 
		 * @param is file handle to the file being parsed
		 * @param rec the container record grouping the spectrum's record
		 * @throws IOException 
		 */
		Spectrum(IRandomAccess is, ContainerRecord rec) throws IOException {
			/*
			 * The record format is a nested container record containing
			 * 
			 * Name - the name of the spectrum
			 * The spectrum container record
			 *    A nesting record
			 *        Wavelengths - the wavelengths of the spectral components
			 *        Magnitudes - the measured signal magnitudes of the fluorophore
			 *        more stuff like calibration
			 * Color - a nested record containing the RGB values for display
			 * Selected - a boolean record that tells whether or not the spectrum is selected
			 * more stuff like acquisition settings 
			 */
			final long oldPos = is.getFilePointer();
			try {
				for (IM3Record subRec:rec.parseChunks(is)) {
					if (subRec.name.equals(FIELD_NAME) && 
						(subRec instanceof StringIM3Record)	) {
						name = ((StringIM3Record)subRec).getValue(is);
					} else if (subRec.name.equals(FIELD_SPECTRUM) &&
							(subRec instanceof ContainerRecord)) {
						parseSpectrumRecord(is, (ContainerRecord)subRec);
					}
				}
			} finally {
				is.seek(oldPos);
			}
		}
		/**
		 * @return the name of this spectrum or null if unnamed
		 */
		public String getName() {
			return name;
		}
		/**
		 * @return the wavelengths for each of the spectral bins
		 */
		public float [] getWavelengths() {
			return wavelengths;
		}
		/**
		 * @return the magnitudes of the signals for each of the spectral bins
		 */
		public float [] getMagnitudes() {
			return magnitudes;
		}
		/**
		 * Parse the spectrum record
		 * 
		 * @param is the file handle
		 * @param subRec the spectrum container record
		 * @throws IOException 
		 */
		private void parseSpectrumRecord(IRandomAccess is, ContainerRecord rec) throws IOException {
			for (IM3Record subRec:rec.parseChunks(is)) {
				if (subRec instanceof ContainerRecord) {
					for (IM3Record subSubRec:((ContainerRecord)subRec).parseChunks(is)) {
						if (subSubRec.name.equals(FIELD_WAVELENGTHS) &&
							(subSubRec instanceof FloatIM3Record)) {
							wavelengths = ((FloatIM3Record)subSubRec).getEntries(is);
						} else if (subSubRec.name.equals(FIELD_MAGNITUDES) &&
								(subSubRec instanceof FloatIM3Record)) {
							magnitudes = ((FloatIM3Record)subSubRec).getEntries(is);
						}
					}
				}
			}
		}
		
	}
	/* (non-Javadoc)
	 * @see loci.formats.FormatReader#setSeries(int)
	 */
	@Override
	public void setSeries(int no) {
		super.setSeries(no);
		data = null;
	}

	/* (non-Javadoc)
	 * @see loci.formats.FormatReader#close(boolean)
	 */
	@Override
	public void close(boolean fileOnly) throws IOException {
		super.close(fileOnly);
		data = null;
	}

	/**
	 * Write a summary of each field in the IM3 file to the writer
	 * @throws IOException
	 */
	public void writeSummary() throws IOException {
		IRandomAccess is = Location.getHandle(getCurrentFile(), false);
		is.setOrder(ByteOrder.LITTLE_ENDIAN);
		for (IM3Record rec: records) {
			rec.writeSummary(is, "");
		}
	}
	/**
	 * Write a summary of each record to STDOUT
	 * 
	 * @param args
	 */
	static public void main(String [] args){
		final IM3Reader reader = new IM3Reader();
		try {
			reader.setId(args[0]);
			reader.writeSummary();
		} catch (FormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}
