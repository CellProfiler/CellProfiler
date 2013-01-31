/*
 * #%L
 * OME SCIFIO package for reading and converting scientific file formats.
 * %%
 * Copyright (C) 2005 - 2012 Open Microscopy Environment:
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
 * 
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of any organization.
 * #L%
 */

package loci.formats.tiff;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Vector;

import loci.common.Constants;
import loci.common.DataTools;
import loci.common.RandomAccessInputStream;
import loci.common.Region;
import loci.common.enumeration.EnumException;
import loci.formats.FormatException;
import loci.formats.codec.BitBuffer;
import loci.formats.codec.CodecOptions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Parses TIFF data from an input source.
 *
 * <dl><dt><b>Source code:</b></dt>
 * <dd><a href="http://trac.openmicroscopy.org.uk/ome/browser/bioformats.git/components/bio-formats/src/loci/formats/tiff/TiffParser.java">Trac</a>,
 * <a href="http://git.openmicroscopy.org/?p=bioformats.git;a=blob;f=components/bio-formats/src/loci/formats/tiff/TiffParser.java;hb=HEAD">Gitweb</a></dd></dl>
 *
 * @author Curtis Rueden ctrueden at wisc.edu
 * @author Eric Kjellman egkjellman at wisc.edu
 * @author Melissa Linkert melissa at glencoesoftware.com
 * @author Chris Allan callan at blackcat.ca
 */
public class TiffParser {

	{
		System.err.println("Using patched TiffParser");
	}
  // -- Constants --

  private static final Logger LOGGER =
    LoggerFactory.getLogger(TiffParser.class);

  // -- Fields --

  /** Input source from which to parse TIFF data. */
  protected RandomAccessInputStream in;

  /** Cached tile buffer to avoid re-allocations when reading tiles. */
  private byte[] cachedTileBuffer;

  /** Whether or not the TIFF file contains BigTIFF data. */
  private boolean bigTiff;

  /** Whether or not 64-bit offsets are used for non-BigTIFF files. */
  private boolean fakeBigTiff = false;

  private boolean ycbcrCorrection = true;

  private boolean equalStrips = false;

  private boolean doCaching;

  /** Cached list of IFDs in the current file. */
  private IFDList ifdList;

  /** Cached first IFD in the current file. */
  private IFD firstIFD;

  private int ifdCount = 0;

  /** Codec options to be used when decoding compressed pixel data. */
  private CodecOptions codecOptions = CodecOptions.getDefaultOptions();

  // -- Constructors --

  /** Constructs a new TIFF parser from the given file name. */
  public TiffParser(String filename) throws IOException {
    this(new RandomAccessInputStream(filename));
  }

  /** Constructs a new TIFF parser from the given input source. */
  public TiffParser(RandomAccessInputStream in) {
    this.in = in;
    doCaching = true;
    try {
      long fp = in.getFilePointer();
      checkHeader();
      in.seek(fp);
    }
    catch (IOException e) { }
  }

  // -- TiffParser methods --

  /**
   * Sets whether or not to assume that strips are of equal size.
   * @param equalStrips Whether or not the strips are of equal size.
   */
  public void setAssumeEqualStrips(boolean equalStrips) {
    this.equalStrips = equalStrips;
  }

  /**
   * Sets the codec options to be used when decompressing pixel data.
   * @param codecOptions Codec options to use.
   */
  public void setCodecOptions(CodecOptions codecOptions) {
    this.codecOptions = codecOptions;
  }

  /**
   * Retrieves the current set of codec options being used to decompress pixel
   * data.
   * @return See above.
   */
  public CodecOptions getCodecOptions() {
    return codecOptions;
  }

  /** Sets whether or not IFD entries should be cached. */
  public void setDoCaching(boolean doCaching) {
    this.doCaching = doCaching;
  }

  /** Sets whether or not 64-bit offsets are used for non-BigTIFF files. */
  public void setUse64BitOffsets(boolean use64Bit) {
    fakeBigTiff = use64Bit;
  }

  /** Sets whether or not YCbCr color correction is allowed. */
  public void setYCbCrCorrection(boolean correctionAllowed) {
    ycbcrCorrection = correctionAllowed;
  }

  /** Gets the stream from which TIFF data is being parsed. */
  public RandomAccessInputStream getStream() {
    return in;
  }

  /** Tests this stream to see if it represents a TIFF file. */
  public boolean isValidHeader() {
    try {
      return checkHeader() != null;
    }
    catch (IOException e) {
      return false;
    }
  }

  /**
   * Checks the TIFF header.
   *
   * @return true if little-endian,
   *         false if big-endian,
   *         or null if not a TIFF.
   */
  public Boolean checkHeader() throws IOException {
    if (in.length() < 4) return null;

    // byte order must be II or MM
    in.seek(0);
    int endianOne = in.read();
    int endianTwo = in.read();
    boolean littleEndian = endianOne == TiffConstants.LITTLE &&
      endianTwo == TiffConstants.LITTLE; // II
    boolean bigEndian = endianOne == TiffConstants.BIG &&
      endianTwo == TiffConstants.BIG; // MM
    if (!littleEndian && !bigEndian) return null;

    // check magic number (42)
    in.order(littleEndian);
    short magic = in.readShort();
    bigTiff = magic == TiffConstants.BIG_TIFF_MAGIC_NUMBER;
    if (magic != TiffConstants.MAGIC_NUMBER &&
      magic != TiffConstants.BIG_TIFF_MAGIC_NUMBER)
    {
      return null;
    }

    return new Boolean(littleEndian);
  }

  /** Returns whether or not the current TIFF file contains BigTIFF data. */
  public boolean isBigTiff() {
    return bigTiff;
  }

  // -- TiffParser methods - IFD parsing --

  /** Returns all IFDs in the file.  */
  public IFDList getIFDs() throws IOException {
    if (ifdList != null) return ifdList;

    long[] offsets = getIFDOffsets();
    IFDList ifds = new IFDList();

    for (long offset : offsets) {
      IFD ifd = getIFD(offset);
      if (ifd == null) continue;
      if (ifd.containsKey(IFD.IMAGE_WIDTH)) ifds.add(ifd);
      long[] subOffsets = null;
      try {
        if (!doCaching && ifd.containsKey(IFD.SUB_IFD)) {
          fillInIFD(ifd);
        }
        subOffsets = ifd.getIFDLongArray(IFD.SUB_IFD);
      }
      catch (FormatException e) { }
      if (subOffsets != null) {
        for (long subOffset : subOffsets) {
          IFD sub = getIFD(subOffset);
          if (sub != null) {
            ifds.add(sub);
          }
        }
      }
    }
    if (doCaching) ifdList = ifds;

    return ifds;
  }

  /** Returns thumbnail IFDs. */
  public IFDList getThumbnailIFDs() throws IOException {
    IFDList ifds = getIFDs();
    IFDList thumbnails = new IFDList();
    for (IFD ifd : ifds) {
      Number subfile = (Number) ifd.getIFDValue(IFD.NEW_SUBFILE_TYPE);
      int subfileType = subfile == null ? 0 : subfile.intValue();
      if (subfileType == 1) {
        thumbnails.add(ifd);
      }
    }
    return thumbnails;
  }

  /** Returns non-thumbnail IFDs. */
  public IFDList getNonThumbnailIFDs() throws IOException {
    IFDList ifds = getIFDs();
    IFDList nonThumbs = new IFDList();
    for (IFD ifd : ifds) {
      Number subfile = (Number) ifd.getIFDValue(IFD.NEW_SUBFILE_TYPE);
      int subfileType = subfile == null ? 0 : subfile.intValue();
      if (subfileType != 1 || ifds.size() <= 1) {
        nonThumbs.add(ifd);
      }
    }
    return nonThumbs;
  }

  /** Returns EXIF IFDs. */
  public IFDList getExifIFDs() throws FormatException, IOException {
    IFDList ifds = getIFDs();
    IFDList exif = new IFDList();
    for (IFD ifd : ifds) {
      long offset = ifd.getIFDLongValue(IFD.EXIF, 0);
      if (offset != 0) {
        IFD exifIFD = getIFD(offset);
        if (exifIFD != null) {
          exif.add(exifIFD);
        }
      }
    }
    return exif;
  }

  /** Gets the offsets to every IFD in the file. */
  public long[] getIFDOffsets() throws IOException {
    // check TIFF header
    int bytesPerEntry = bigTiff ? TiffConstants.BIG_TIFF_BYTES_PER_ENTRY :
      TiffConstants.BYTES_PER_ENTRY;

    Vector<Long> offsets = new Vector<Long>();
    long offset = getFirstOffset();
    while (offset > 0 && offset < in.length()) {
      in.seek(offset);
      offsets.add(offset);
      int nEntries = bigTiff ? (int) in.readLong() : in.readUnsignedShort();
      in.skipBytes(nEntries * bytesPerEntry);
      offset = getNextOffset(offset);
    }

    long[] f = new long[offsets.size()];
    for (int i=0; i<f.length; i++) {
      f[i] = offsets.get(i).longValue();
    }
    ifdCount = f.length;

    return f;
  }

  /**
   * Gets the first IFD within the TIFF file, or null
   * if the input source is not a valid TIFF file.
   */
  public IFD getFirstIFD() throws IOException {
    if (firstIFD != null) return firstIFD;
    long offset = getFirstOffset();
    IFD ifd = getIFD(offset);
    if (doCaching) firstIFD = ifd;
    return ifd;
  }

  /**
   * Retrieve a given entry from the first IFD in the stream.
   *
   * @param tag the tag of the entry to be retrieved.
   * @return an object representing the entry's fields.
   * @throws IOException when there is an error accessing the stream.
   * @throws IllegalArgumentException when the tag number is unknown.
   */
  // TODO : Try to remove this method.  It is only being used by
  //        loci.formats.in.MetamorphReader.
  public TiffIFDEntry getFirstIFDEntry(int tag) throws IOException {
    // Get the offset of the first IFD
    long offset = getFirstOffset();
    if (offset < 0) return null;

    // The following loosely resembles the logic of getIFD()...
    in.seek(offset);
    long numEntries = bigTiff ? in.readLong() : in.readUnsignedShort();

    for (int i = 0; i < numEntries; i++) {
      in.seek(offset + // The beginning of the IFD
        (bigTiff ? 8 : 2) + // The width of the initial numEntries field
        (bigTiff ? TiffConstants.BIG_TIFF_BYTES_PER_ENTRY :
        TiffConstants.BYTES_PER_ENTRY) * i);

      TiffIFDEntry entry = readTiffIFDEntry();
      if (entry.getTag() == tag) {
        return entry;
      }
    }
    throw new IllegalArgumentException("Unknown tag: " + tag);
  }

  /**
   * Gets offset to the first IFD, or -1 if stream is not TIFF.
   */
  public long getFirstOffset() throws IOException {
    Boolean header = checkHeader();
    if (header == null) return -1;
    if (bigTiff) in.skipBytes(4);
    return getNextOffset(0);
  }

  /** Gets the IFD stored at the given offset.  */
  public IFD getIFD(long offset) throws IOException {
    if (offset < 0 || offset >= in.length()) return null;
    IFD ifd = new IFD();

    // save little-endian flag to internal LITTLE_ENDIAN tag
    ifd.put(new Integer(IFD.LITTLE_ENDIAN), new Boolean(in.isLittleEndian()));
    ifd.put(new Integer(IFD.BIG_TIFF), new Boolean(bigTiff));

    // read in directory entries for this IFD
    LOGGER.trace("getIFDs: seeking IFD at {}", offset);
    in.seek(offset);
    long numEntries = bigTiff ? in.readLong() : in.readUnsignedShort();
    LOGGER.trace("getIFDs: {} directory entries to read", numEntries);
    if (numEntries == 0 || numEntries == 1) return ifd;

    int bytesPerEntry = bigTiff ?
      TiffConstants.BIG_TIFF_BYTES_PER_ENTRY : TiffConstants.BYTES_PER_ENTRY;
    int baseOffset = bigTiff ? 8 : 2;

    for (int i=0; i<numEntries; i++) {
      in.seek(offset + baseOffset + bytesPerEntry * i);

      TiffIFDEntry entry = null;
      try {
        entry = readTiffIFDEntry();
      }
      catch (EnumException e) {
        LOGGER.debug("", e);
      }
      if (entry == null) break;
      int count = entry.getValueCount();
      int tag = entry.getTag();
      long pointer = entry.getValueOffset();
      int bpe = entry.getType().getBytesPerElement();

      if (count < 0 || bpe <= 0) {
        // invalid data
        in.skipBytes(bytesPerEntry - 4 - (bigTiff ? 8 : 4));
        continue;
      }
      Object value = null;

      long inputLen = in.length();
      if (count * bpe + pointer > inputLen) {
        int oldCount = count;
        count = (int) ((inputLen - pointer) / bpe);
        LOGGER.trace("getIFDs: truncated {} array elements for tag {}",
          (oldCount - count), tag);
        if (count < 0) count = oldCount;
      }
      if (count < 0 || count > in.length()) break;

      if (pointer != in.getFilePointer() && !doCaching) {
        value = entry;
      }
      else value = getIFDValue(entry);

      if (value != null && !ifd.containsKey(new Integer(tag))) {
        ifd.put(new Integer(tag), value);
      }
    }

    in.seek(offset + baseOffset + bytesPerEntry * numEntries);

    return ifd;
  }

  /** Fill in IFD entries that are stored at an arbitrary offset. */
  public void fillInIFD(IFD ifd) throws IOException {
    HashSet<TiffIFDEntry> entries = new HashSet<TiffIFDEntry>();
    for (Object key : ifd.keySet()) {
      if (ifd.get(key) instanceof TiffIFDEntry) {
        entries.add((TiffIFDEntry) ifd.get(key));
      }
    }

    for (TiffIFDEntry entry : entries) {
      if (entry.getValueCount() < 10 * 1024 * 1024 || entry.getTag() < 32768) {
        ifd.put(new Integer(entry.getTag()), getIFDValue(entry));
      }
    }
  }

  /** Retrieve the value corresponding to the given TiffIFDEntry. */
  public Object getIFDValue(TiffIFDEntry entry) throws IOException {
    IFDType type = entry.getType();
    int count = entry.getValueCount();
    long offset = entry.getValueOffset();

    LOGGER.trace("Reading entry {} from {}; type={}, count={}",
      new Object[] {entry.getTag(), offset, type, count});

    if (offset >= in.length()) {
      return null;
    }

    if (offset != in.getFilePointer()) {
      in.seek(offset);
    }

    if (type == IFDType.BYTE) {
      // 8-bit unsigned integer
      if (count == 1) return new Short(in.readByte());
      byte[] bytes = new byte[count];
      in.readFully(bytes);
      // bytes are unsigned, so use shorts
      short[] shorts = new short[count];
      for (int j=0; j<count; j++) shorts[j] = (short) (bytes[j] & 0xff);
      return shorts;
    }
    else if (type == IFDType.ASCII) {
      // 8-bit byte that contain a 7-bit ASCII code;
      // the last byte must be NUL (binary zero)
      byte[] ascii = new byte[count];
      in.read(ascii);

      // count number of null terminators
      int nullCount = 0;
      for (int j=0; j<count; j++) {
        if (ascii[j] == 0 || j == count - 1) nullCount++;
      }

      // convert character array to array of strings
      String[] strings = nullCount == 1 ? null : new String[nullCount];
      String s = null;
      int c = 0, ndx = -1;
      for (int j=0; j<count; j++) {
        if (ascii[j] == 0) {
          s = new String(ascii, ndx + 1, j - ndx - 1, Constants.ENCODING);
          ndx = j;
        }
        else if (j == count - 1) {
          // handle non-null-terminated strings
          s = new String(ascii, ndx + 1, j - ndx, Constants.ENCODING);
        }
        else s = null;
        if (strings != null && s != null) strings[c++] = s;
      }
      return strings == null ? (Object) s : strings;
    }
    else if (type == IFDType.SHORT) {
      // 16-bit (2-byte) unsigned integer
      if (count == 1) return new Integer(in.readUnsignedShort());
      int[] shorts = new int[count];
      for (int j=0; j<count; j++) {
        shorts[j] = in.readUnsignedShort();
      }
      return shorts;
    }
    else if (type == IFDType.LONG || type == IFDType.IFD) {
      // 32-bit (4-byte) unsigned integer
      if (count == 1) return new Long(in.readInt());
      long[] longs = new long[count];
      for (int j=0; j<count; j++) {
        if (in.getFilePointer() + 4 <= in.length()) {
          longs[j] = in.readInt();
        }
      }
      return longs;
    }
    else if (type == IFDType.LONG8 || type == IFDType.SLONG8
             || type == IFDType.IFD8) {
      if (count == 1) return new Long(in.readLong());
      long[] longs = null;

      if (equalStrips && (entry.getTag() == IFD.STRIP_BYTE_COUNTS ||
        entry.getTag() == IFD.TILE_BYTE_COUNTS))
      {
        longs = new long[1];
        longs[0] = in.readLong();
      }
      else if (equalStrips && (entry.getTag() == IFD.STRIP_OFFSETS ||
        entry.getTag() == IFD.TILE_OFFSETS))
      {
        OnDemandLongArray offsets = new OnDemandLongArray(in);
        offsets.setSize(count);
        return offsets;
      }
      else {
        longs = new long[count];
        for (int j=0; j<count; j++) longs[j] = in.readLong();
      }
      return longs;
    }
    else if (type == IFDType.RATIONAL || type == IFDType.SRATIONAL) {
      // Two LONGs or SLONGs: the first represents the numerator
      // of a fraction; the second, the denominator
      if (count == 1) return new TiffRational(in.readInt(), in.readInt());
      TiffRational[] rationals = new TiffRational[count];
      for (int j=0; j<count; j++) {
        rationals[j] = new TiffRational(in.readInt(), in.readInt());
      }
      return rationals;
    }
    else if (type == IFDType.SBYTE || type == IFDType.UNDEFINED) {
      // SBYTE: An 8-bit signed (twos-complement) integer
      // UNDEFINED: An 8-bit byte that may contain anything,
      // depending on the definition of the field
      if (count == 1) return new Byte(in.readByte());
      byte[] sbytes = new byte[count];
      in.read(sbytes);
      return sbytes;
    }
    else if (type == IFDType.SSHORT) {
      // A 16-bit (2-byte) signed (twos-complement) integer
      if (count == 1) return new Short(in.readShort());
      short[] sshorts = new short[count];
      for (int j=0; j<count; j++) sshorts[j] = in.readShort();
      return sshorts;
    }
    else if (type == IFDType.SLONG) {
      // A 32-bit (4-byte) signed (twos-complement) integer
      if (count == 1) return new Integer(in.readInt());
      int[] slongs = new int[count];
      for (int j=0; j<count; j++) slongs[j] = in.readInt();
      return slongs;
    }
    else if (type == IFDType.FLOAT) {
      // Single precision (4-byte) IEEE format
      if (count == 1) return new Float(in.readFloat());
      float[] floats = new float[count];
      for (int j=0; j<count; j++) floats[j] = in.readFloat();
      return floats;
    }
    else if (type == IFDType.DOUBLE) {
      // Double precision (8-byte) IEEE format
      if (count == 1) return new Double(in.readDouble());
      double[] doubles = new double[count];
      for (int j=0; j<count; j++) {
        doubles[j] = in.readDouble();
      }
      return doubles;
    }

    return null;
  }

  /** Convenience method for obtaining a stream's first ImageDescription. */
  public String getComment() throws IOException {
    IFD firstIFD = getFirstIFD();
    if (firstIFD == null) {
      return null;
    }
    fillInIFD(firstIFD);
    return firstIFD.getComment();
  }

  // -- TiffParser methods - image reading --

  public byte[] getTile(IFD ifd, byte[] buf, int row, int col)
    throws FormatException, IOException
  {
    byte[] jpegTable = (byte[]) ifd.getIFDValue(IFD.JPEG_TABLES);

    codecOptions.interleaved = true;
    codecOptions.littleEndian = ifd.isLittleEndian();

    long tileWidth = ifd.getTileWidth();
    long tileLength = ifd.getTileLength();
    int samplesPerPixel = ifd.getSamplesPerPixel();
    int planarConfig = ifd.getPlanarConfiguration();
    TiffCompression compression = ifd.getCompression();

    long numTileCols = ifd.getTilesPerRow();

    int pixel = ifd.getBytesPerSample()[0];
    int effectiveChannels = planarConfig == 2 ? 1 : samplesPerPixel;

    long[] stripByteCounts = ifd.getStripByteCounts();
    long[] rowsPerStrip = ifd.getRowsPerStrip();

    int offsetIndex = (int) (row * numTileCols + col);
    int countIndex = offsetIndex;
    if (equalStrips) {
      countIndex = 0;
    }
    if (stripByteCounts[countIndex] == (rowsPerStrip[0] * tileWidth) &&
      pixel > 1)
    {
      stripByteCounts[countIndex] *= pixel;
    }

    long stripOffset = 0;
    long nStrips = 0;

    if (ifd.getOnDemandStripOffsets() != null) {
      OnDemandLongArray stripOffsets = ifd.getOnDemandStripOffsets();
      stripOffset = stripOffsets.get(offsetIndex);
      nStrips = stripOffsets.size();
    }
    else {
      long[] stripOffsets = ifd.getStripOffsets();
      stripOffset = stripOffsets[offsetIndex];
      nStrips = stripOffsets.length;
    }

    int size = (int) (tileWidth * tileLength * pixel * effectiveChannels);

    if (buf == null) buf = new byte[size];
    if (stripByteCounts[countIndex] == 0 || stripOffset >= in.length()) {
      return buf;
    }
    byte[] tile = new byte[(int) stripByteCounts[countIndex]];

    LOGGER.debug("Reading tile Length {} Offset {}", tile.length, stripOffset);
    in.seek(stripOffset);
    in.read(tile);

    codecOptions.maxBytes = (int) Math.max(size, tile.length);
    codecOptions.ycbcr =
      ifd.getPhotometricInterpretation() == PhotoInterp.Y_CB_CR &&
      ifd.getIFDIntValue(IFD.Y_CB_CR_SUB_SAMPLING) == 1 && ycbcrCorrection;

    if (jpegTable != null) {
      byte[] q = new byte[jpegTable.length + tile.length - 4];
      System.arraycopy(jpegTable, 0, q, 0, jpegTable.length - 2);
      System.arraycopy(tile, 2, q, jpegTable.length - 2, tile.length - 2);
      tile = compression.decompress(q, codecOptions);
    }
    else tile = compression.decompress(tile, codecOptions);
    TiffCompression.undifference(tile, ifd);
    unpackBytes(buf, 0, tile, ifd);

    if (planarConfig == 2 && !ifd.isTiled() && ifd.getSamplesPerPixel() > 1) {
      int channel = (int) (row % nStrips);
      if (channel < ifd.getBytesPerSample().length) {
        int realBytes = ifd.getBytesPerSample()[channel];
        if (realBytes != pixel) {
          // re-pack pixels to account for differing bits per sample

          boolean littleEndian = ifd.isLittleEndian();
          int[] samples = new int[buf.length / pixel];
          for (int i=0; i<samples.length; i++) {
            samples[i] =
              DataTools.bytesToInt(buf, i * realBytes, realBytes, littleEndian);
          }

          for (int i=0; i<samples.length; i++) {
            DataTools.unpackBytes(
              samples[i], buf, i * pixel, pixel, littleEndian);
          }
        }
      }
    }

    return buf;
  }

  public byte[] getSamples(IFD ifd, byte[] buf)
    throws FormatException, IOException
  {
    long width = ifd.getImageWidth();
    long length = ifd.getImageLength();
    return getSamples(ifd, buf, 0, 0, width, length);
  }

  public byte[] getSamples(IFD ifd, byte[] buf, int x, int y,
    long width, long height) throws FormatException, IOException
  {
    return getSamples(ifd, buf, x, y, width, height, 0, 0);
  }

  public byte[] getSamples(IFD ifd, byte[] buf, int x, int y,
    long width, long height, int overlapX, int overlapY)
    throws FormatException, IOException
  {
    LOGGER.trace("parsing IFD entries");

    // get internal non-IFD entries
    boolean littleEndian = ifd.isLittleEndian();
    in.order(littleEndian);

    // get relevant IFD entries
    int samplesPerPixel = ifd.getSamplesPerPixel();
    long tileWidth = ifd.getTileWidth();
    long tileLength = ifd.getTileLength();
    if (tileLength <= 0) {
      LOGGER.trace("Tile length is {}; setting it to {}", tileLength, height);
      tileLength = height;
    }

    long numTileRows = ifd.getTilesPerColumn();
    long numTileCols = ifd.getTilesPerRow();

    PhotoInterp photoInterp = ifd.getPhotometricInterpretation();
    int planarConfig = ifd.getPlanarConfiguration();
    int pixel = ifd.getBytesPerSample()[0];
    int effectiveChannels = planarConfig == 2 ? 1 : samplesPerPixel;

    if (LOGGER.isTraceEnabled()) {
      ifd.printIFD();
    }

    if (width * height > Integer.MAX_VALUE) {
      throw new FormatException("Sorry, ImageWidth x ImageLength > " +
        Integer.MAX_VALUE + " is not supported (" +
        width + " x " + height + ")");
    }
    if (width * height * effectiveChannels * pixel > Integer.MAX_VALUE) {
      throw new FormatException("Sorry, ImageWidth x ImageLength x " +
        "SamplesPerPixel x BitsPerSample > " + Integer.MAX_VALUE +
        " is not supported (" + width + " x " + height + " x " +
        samplesPerPixel + " x " + (pixel * 8) + ")");
    }

    // casting to int is safe because we have already determined that
    // width * height is less than Integer.MAX_VALUE
    int numSamples = (int) (width * height);

    // read in image strips
    LOGGER.trace("reading image data (samplesPerPixel={}; numSamples={})",
      samplesPerPixel, numSamples);

    TiffCompression compression = ifd.getCompression();

    if (compression == TiffCompression.JPEG_2000 ||
      compression == TiffCompression.JPEG_2000_LOSSY)
    {
      codecOptions = compression.getCompressionCodecOptions(ifd, codecOptions);
    }
    else codecOptions = compression.getCompressionCodecOptions(ifd);
    codecOptions.interleaved = true;
    codecOptions.littleEndian = ifd.isLittleEndian();
    long imageLength = ifd.getImageLength();

    // special case: if we only need one tile, and that tile doesn't need
    // any special handling, then we can just read it directly and return
    if ((x % tileWidth) == 0 && (y % tileLength) == 0 && width == tileWidth &&
      height == imageLength && samplesPerPixel == 1 &&
      (ifd.getBitsPerSample()[0] % 8) == 0 &&
      photoInterp != PhotoInterp.WHITE_IS_ZERO &&
      photoInterp != PhotoInterp.CMYK && photoInterp != PhotoInterp.Y_CB_CR &&
      compression == TiffCompression.UNCOMPRESSED)
    {
      long[] stripOffsets = ifd.getStripOffsets();
      long[] stripByteCounts = ifd.getStripByteCounts();

      if (stripOffsets != null && stripByteCounts != null) {
        long column = x / tileWidth;
        int firstTile = (int) ((y / tileLength) * numTileCols + column);
        int lastTile =
          (int) (((y + height) / tileLength) * numTileCols + column);
        lastTile = (int) Math.min(lastTile, stripOffsets.length - 1);

        int offset = 0;
        for (int tile=firstTile; tile<=lastTile; tile++) {
          long byteCount =
            equalStrips ? stripByteCounts[0] : stripByteCounts[tile];
          if (byteCount == numSamples && pixel > 1) {
            byteCount *= pixel;
          }

          in.seek(stripOffsets[tile]);
          int len = (int) Math.min(buf.length - offset, byteCount);
          in.read(buf, offset, len);
          offset += len;
        }
      }
      return buf;
    }

    long nrows = numTileRows;
    if (planarConfig == 2) numTileRows *= samplesPerPixel;

    Region imageBounds = new Region(x, y, (int) width, (int) height);

    int endX = (int) width + x;
    int endY = (int) height + y;

    long w = tileWidth;
    long h = tileLength;
    int rowLen = pixel * (int) w;//tileWidth;
    int tileSize = (int) (rowLen * h);//tileLength);

    int planeSize = (int) (width * height * pixel);
    int outputRowLen = (int) (pixel * width);

    int bufferSizeSamplesPerPixel = samplesPerPixel;
    if (ifd.getPlanarConfiguration() == 2) bufferSizeSamplesPerPixel = 1;
    int bpp = ifd.getBytesPerSample()[0];
    int bufferSize = (int) tileWidth * (int) tileLength *
      bufferSizeSamplesPerPixel * bpp;

    cachedTileBuffer = new byte[bufferSize];

    Region tileBounds = new Region(0, 0, (int) tileWidth, (int) tileLength);

    for (int row=0; row<numTileRows; row++) {
      for (int col=0; col<numTileCols; col++) {
        tileBounds.x = col * (int) (tileWidth - overlapX);
        tileBounds.y = row * (int) (tileLength - overlapY);

        if (planarConfig == 2) {
          tileBounds.y = (int) ((row % nrows) * (tileLength - overlapY));
        }

        if (!imageBounds.intersects(tileBounds)) continue;

        getTile(ifd, cachedTileBuffer, row, col);

        // adjust tile bounds, if necessary

        int tileX = (int) Math.max(tileBounds.x, x);
        int tileY = (int) Math.max(tileBounds.y, y);
        int realX = tileX % (int) (tileWidth - overlapX);
        int realY = tileY % (int) (tileLength - overlapY);

        int twidth = (int) Math.min(endX - tileX, tileWidth - realX);
        if (twidth <= 0) {
          twidth = (int) Math.max(endX - tileX, tileWidth - realX);
        }
        int theight = (int) Math.min(endY - tileY, tileLength - realY);
        if (theight <= 0) {
          theight = (int) Math.max(endY - tileY, tileLength - realY);
        }
        // copy appropriate portion of the tile to the output buffer

        int copy = pixel * twidth;

        realX *= pixel;
        realY *= rowLen;

        for (int q=0; q<effectiveChannels; q++) {
          int src = (int) (q * tileSize) + realX + realY;
          int dest = (int) (q * planeSize) + pixel * (tileX - x) +
            outputRowLen * (tileY - y);
          if (planarConfig == 2) dest += (planeSize * (row / nrows));

          if (rowLen == outputRowLen) {
            System.arraycopy(cachedTileBuffer, src, buf, dest, copy * theight);
          }
          else {
            for (int tileRow=0; tileRow<theight; tileRow++) {
              System.arraycopy(cachedTileBuffer, src, buf, dest, copy);
              src += rowLen;
              dest += outputRowLen;
            }
          }
        }
      }
    }

    return buf;
  }

  // -- Utility methods - byte stream decoding --

  /**
   * Extracts pixel information from the given byte array according to the
   * bits per sample, photometric interpretation and color map IFD directory
   * entry values, and the specified byte ordering.
   * No error checking is performed.
   */
  public static void unpackBytes(byte[] samples, int startIndex, byte[] bytes,
    IFD ifd) throws FormatException
  {
    boolean planar = ifd.getPlanarConfiguration() == 2;

    TiffCompression compression = ifd.getCompression();
    PhotoInterp photoInterp = ifd.getPhotometricInterpretation();
    if (compression == TiffCompression.JPEG) photoInterp = PhotoInterp.RGB;

    int[] bitsPerSample = ifd.getBitsPerSample();
    int nChannels = bitsPerSample.length;

    int sampleCount = (int) (((long) 8 * bytes.length) / bitsPerSample[0]);
    if (photoInterp == PhotoInterp.Y_CB_CR) sampleCount *= 3;
    if (planar) {
      nChannels = 1;
    }
    else {
      sampleCount /= nChannels;
    }

    LOGGER.trace(
      "unpacking {} samples (startIndex={}; totalBits={}; numBytes={})",
      new Object[] {sampleCount, startIndex, nChannels * bitsPerSample[0],
      bytes.length});

    long imageWidth = ifd.getImageWidth();
    long imageHeight = ifd.getImageLength();

    int bps0 = bitsPerSample[0];
    int numBytes = ifd.getBytesPerSample()[0];
    int nSamples = samples.length / (nChannels * numBytes);

    boolean noDiv8 = bps0 % 8 != 0;
    boolean bps8 = bps0 == 8;
    boolean bps16 = bps0 == 16;

    boolean littleEndian = ifd.isLittleEndian();

    BitBuffer bb = new BitBuffer(bytes);

    // Hyper optimisation that takes any 8-bit or 16-bit data, where there is
    // only one channel, the source byte buffer's size is less than or equal to
    // that of the destination buffer and for which no special unpacking is
    // required and performs a simple array copy. Over the course of reading
    // semi-large datasets this can save **billions** of method calls.
    // Wed Aug  5 19:04:59 BST 2009
    // Chris Allan <callan@glencoesoftware.com>
    if ((bps8 || bps16) && bytes.length <= samples.length && nChannels == 1
        && photoInterp != PhotoInterp.WHITE_IS_ZERO
        && photoInterp != PhotoInterp.CMYK
        && photoInterp != PhotoInterp.Y_CB_CR) {
      System.arraycopy(bytes, 0, samples, 0, bytes.length);
      return;
    }

    long maxValue = (long) Math.pow(2, bps0) - 1;
    if (photoInterp == PhotoInterp.CMYK) maxValue = Integer.MAX_VALUE;

    int skipBits = (int) (8 - ((imageWidth * bps0 * nChannels) % 8));
    if (skipBits == 8 ||
      (bytes.length * 8 < bps0 * (nChannels * imageWidth + imageHeight)))
    {
      skipBits = 0;
    }

    // set up YCbCr-specific values
    float lumaRed = PhotoInterp.LUMA_RED;
    float lumaGreen = PhotoInterp.LUMA_GREEN;
    float lumaBlue = PhotoInterp.LUMA_BLUE;
    int[] reference = ifd.getIFDIntArray(IFD.REFERENCE_BLACK_WHITE);
    if (reference == null) {
      reference = new int[] {0, 0, 0, 0, 0, 0};
    }
    int[] subsampling = ifd.getIFDIntArray(IFD.Y_CB_CR_SUB_SAMPLING);
    TiffRational[] coefficients = (TiffRational[])
      ifd.getIFDValue(IFD.Y_CB_CR_COEFFICIENTS);
    if (coefficients != null) {
      lumaRed = coefficients[0].floatValue();
      lumaGreen = coefficients[1].floatValue();
      lumaBlue = coefficients[2].floatValue();
    }
    int subX = subsampling == null ? 2 : subsampling[0];
    int subY = subsampling == null ? 2 : subsampling[1];
    int block = subX * subY;
    int nTiles = (int) (imageWidth / subX);

    // unpack pixels
    for (int sample=0; sample<sampleCount; sample++) {
      int ndx = startIndex + sample;
      if (ndx >= nSamples) break;

      for (int channel=0; channel<nChannels; channel++) {
        int index = numBytes * (sample * nChannels + channel);
        int outputIndex = (channel * nSamples + ndx) * numBytes;

        // unpack non-YCbCr samples
        if (photoInterp != PhotoInterp.Y_CB_CR) {
          long value = 0;

          if (noDiv8) {
            // bits per sample is not a multiple of 8

            if ((channel == 0 && photoInterp == PhotoInterp.RGB_PALETTE) ||
              (photoInterp != PhotoInterp.CFA_ARRAY &&
              photoInterp != PhotoInterp.RGB_PALETTE))
            {
              value = bb.getBits(bps0) & 0xffff;
              if ((ndx % imageWidth) == imageWidth - 1) {
                bb.skipBits(skipBits);
              }
            }
          }
          else {
            value = DataTools.bytesToLong(bytes, index, numBytes, littleEndian);
          }

          if (photoInterp == PhotoInterp.WHITE_IS_ZERO ||
            photoInterp == PhotoInterp.CMYK)
          {
            value = maxValue - value;
          }

          if (outputIndex + numBytes <= samples.length) {
            DataTools.unpackBytes(value, samples, outputIndex, numBytes,
              littleEndian);
          }
        }
        else {
          // unpack YCbCr samples; these need special handling, as each of
          // the RGB components depends upon two or more of the YCbCr components
          if (channel == nChannels - 1) {
            int lumaIndex = sample + (2 * (sample / block));
            int chromaIndex = (sample / block) * (block + 2) + block;

            if (chromaIndex + 1 >= bytes.length) break;

            int tile = ndx / block;
            int pixel = ndx % block;
            long r = subY * (tile / nTiles) + (pixel / subX);
            long c = subX * (tile % nTiles) + (pixel % subX);

            int idx = (int) (r * imageWidth + c);

            if (idx < nSamples) {
              int y = (bytes[lumaIndex] & 0xff) - reference[0];
              int cb = (bytes[chromaIndex] & 0xff) - reference[2];
              int cr = (bytes[chromaIndex + 1] & 0xff) - reference[4];

              int red = (int) (cr * (2 - 2 * lumaRed) + y);
              int blue = (int) (cb * (2 - 2 * lumaBlue) + y);
              int green = (int)
                ((y - lumaBlue * blue - lumaRed * red) / lumaGreen);

              samples[idx] = (byte) (red & 0xff);
              samples[nSamples + idx] = (byte) (green & 0xff);
              samples[2*nSamples + idx] = (byte) (blue & 0xff);
            }
          }
        }
      }
    }
  }

  /**
   * Read a file offset.
   * For bigTiff, a 64-bit number is read.  For other Tiffs, a 32-bit number
   * is read and possibly adjusted for a possible carry-over from the previous
   * offset.
   */
  long getNextOffset(long previous) throws IOException {
    if (bigTiff || fakeBigTiff) {
      return in.readLong();
    }
    long offset = (previous & ~0xffffffffL) | (in.readInt() & 0xffffffffL);

    // Only adjust the offset if we know that the file is too large for 32-bit
    // offsets to be accurate; otherwise, we're making the incorrect assumption
    // that IFDs are stored sequentially.
    if (offset < previous && offset != 0 && in.length() > Integer.MAX_VALUE) {
      offset += 0x100000000L;
    }
    return offset;
  }

  TiffIFDEntry readTiffIFDEntry() throws IOException {
    int entryTag = in.readUnsignedShort();

    // Parse the entry's "Type"
    IFDType entryType;
    try {
       entryType = IFDType.get(in.readUnsignedShort());
    }
    catch (EnumException e) {
      LOGGER.error("Error reading IFD type at: {}", in.getFilePointer());
      throw e;
    }

    // Parse the entry's "ValueCount"
    int valueCount = bigTiff ? (int) in.readLong() : in.readInt();
    if (valueCount < 0) {
      throw new RuntimeException("Count of '" + valueCount + "' unexpected.");
    }

    int nValueBytes = valueCount * entryType.getBytesPerElement();
    int threshhold = bigTiff ? 8 : 4;
    long offset = nValueBytes > threshhold ?
      getNextOffset(0) : in.getFilePointer();

    return new TiffIFDEntry(entryTag, entryType, valueCount, offset);
  }

}
