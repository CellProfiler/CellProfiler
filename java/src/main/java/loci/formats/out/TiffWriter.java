/*
 * #%L
 * OME SCIFIO package for reading and converting scientific file formats.
 * %%
 * Copyright (C) 2005 - 2013 Open Microscopy Environment:
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

package loci.formats.out;

import java.io.IOException;

import loci.common.RandomAccessInputStream;
import loci.common.RandomAccessOutputStream;
import loci.formats.FormatException;
import loci.formats.FormatTools;
import loci.formats.FormatWriter;
import loci.formats.ImageTools;
import loci.formats.codec.CompressionType;
import loci.formats.gui.AWTImageTools;
import loci.formats.meta.MetadataRetrieve;
import loci.formats.tiff.IFD;
import loci.formats.tiff.TiffCompression;
import loci.formats.tiff.TiffParser;
import loci.formats.tiff.TiffRational;
import loci.formats.tiff.TiffSaver;

import ome.xml.model.primitives.PositiveFloat;

/**
 * TiffWriter is the file format writer for TIFF files.
 *
 * <dl><dt><b>Source code:</b></dt>
 * <dd><a href="http://trac.openmicroscopy.org.uk/ome/browser/bioformats.git/components/bio-formats/src/loci/formats/out/TiffWriter.java">Trac</a>,
 * <a href="http://git.openmicroscopy.org/?p=bioformats.git;a=blob;f=components/bio-formats/src/loci/formats/out/TiffWriter.java;hb=HEAD">Gitweb</a></dd></dl>
 */
public class TiffWriter extends FormatWriter {

  // -- Constants --

  public static final String COMPRESSION_UNCOMPRESSED =
    CompressionType.UNCOMPRESSED.getCompression();
  public static final String COMPRESSION_LZW =
    CompressionType.LZW.getCompression();
  public static final String COMPRESSION_J2K =
    CompressionType.J2K.getCompression();
  public static final String COMPRESSION_J2K_LOSSY =
    CompressionType.J2K_LOSSY.getCompression();
  public static final String COMPRESSION_JPEG =
    CompressionType.JPEG.getCompression();

  // -- Fields --

  /** Whether or not the output file is a BigTIFF file. */
  protected boolean isBigTiff;

  /** The TiffSaver that will do most of the writing. */
  protected TiffSaver tiffSaver;

  /** Input stream to use when overwriting data. */
  protected RandomAccessInputStream in;

  /** Whether or not to check the parameters passed to saveBytes. */
  private boolean checkParams = true;

  /**
   * Sets the compression code for the specified IFD.
   * 
   * @param ifd The IFD table to handle.
   */
  private void formatCompression(IFD ifd)
    throws FormatException
  {
    if (compression == null) compression = "";
    TiffCompression compressType = TiffCompression.UNCOMPRESSED;
    if (compression.equals(COMPRESSION_LZW)) {
      compressType = TiffCompression.LZW;
    }
    else if (compression.equals(COMPRESSION_J2K)) {
      compressType = TiffCompression.JPEG_2000;
    }
    else if (compression.equals(COMPRESSION_J2K_LOSSY)) {
      compressType = TiffCompression.JPEG_2000_LOSSY;
    }
    else if (compression.equals(COMPRESSION_JPEG)) {
      compressType = TiffCompression.JPEG;
    }
    Object v = ifd.get(new Integer(IFD.COMPRESSION));
    if (v == null)
      ifd.put(new Integer(IFD.COMPRESSION), compressType.getCode());
  }

  // -- Constructors --

  public TiffWriter() {
    this("Tagged Image File Format", new String[] {"tif", "tiff"});
  }

  public TiffWriter(String format, String[] exts) {
    super(format, exts);
    compressionTypes = new String[] {
      COMPRESSION_UNCOMPRESSED,
      COMPRESSION_LZW,
      COMPRESSION_J2K,
      COMPRESSION_J2K_LOSSY,
      COMPRESSION_JPEG
    };
    isBigTiff = false;
  }

  // -- IFormatHandler API methods --

  /* @see loci.formats.IFormatHandler#setId(String) */
  public void setId(String id) throws FormatException, IOException {
    super.setId(id);

    synchronized (this) {
      setupTiffSaver();
    }
  }

  // -- TiffWriter API methods --

  /**
   * Saves the given image to the specified (possibly already open) file.
   * The IFD hashtable allows specification of TIFF parameters such as bit
   * depth, compression and units.
   */
  public void saveBytes(int no, byte[] buf, IFD ifd)
    throws IOException, FormatException
  {
    MetadataRetrieve r = getMetadataRetrieve();
    int w = r.getPixelsSizeX(series).getValue().intValue();
    int h = r.getPixelsSizeY(series).getValue().intValue();
    saveBytes(no, buf, ifd, 0, 0, w, h);
  }

  /**
   * Saves the given image to the specified series in the current file.
   * The IFD hashtable allows specification of TIFF parameters such as bit
   * depth, compression and units.
   */
  public void saveBytes(int no, byte[] buf, IFD ifd, int x, int y, int w, int h)
    throws IOException, FormatException
  {
    if (checkParams) checkParams(no, buf, x, y, w, h);
    if (ifd == null) ifd = new IFD();
    MetadataRetrieve retrieve = getMetadataRetrieve();
    int type = FormatTools.pixelTypeFromString(
        retrieve.getPixelsType(series).toString());
    int index = no;
    // This operation is synchronized
    synchronized (this) {
      // This operation is synchronized against the TIFF saver.
      synchronized (tiffSaver) {
        index = prepareToWriteImage(no, buf, ifd, x, y, w, h);
        if (index == -1) {
          return;
        }
      }
    }

    tiffSaver.writeImage(buf, ifd, index, type, x, y, w, h,
      no == getPlaneCount() - 1 && getSeries() == retrieve.getImageCount() - 1);
  }

  /**
   * Performs the preparation for work prior to the usage of the TIFF saver.
   * This method is factored out from <code>saveBytes()</code> in an attempt to
   * ensure thread safety.
   */
  private int prepareToWriteImage(
      int no, byte[] buf, IFD ifd, int x, int y, int w, int h)
  throws IOException, FormatException {
    MetadataRetrieve retrieve = getMetadataRetrieve();
    Boolean bigEndian = retrieve.getPixelsBinDataBigEndian(series, 0);
    boolean littleEndian = bigEndian == null ?
      false : !bigEndian.booleanValue();

    // Ensure that no more than one thread manipulated the initialized array
    // at one time.
    synchronized (this) {
      if (no < initialized[series].length && !initialized[series][no]) {
        initialized[series][no] = true;

        RandomAccessInputStream tmp = new RandomAccessInputStream(currentId);
        if (tmp.length() == 0) {
          synchronized (this) {
            // write TIFF header
            tiffSaver.writeHeader();
          }
        }
        tmp.close();
      }
    }

    int c = getSamplesPerPixel();
    int type = FormatTools.pixelTypeFromString(
      retrieve.getPixelsType(series).toString());
    int bytesPerPixel = FormatTools.getBytesPerPixel(type);

    int blockSize = w * h * c * bytesPerPixel;
    if (blockSize > buf.length) {
      c = buf.length / (w * h * bytesPerPixel);
    }

    if (bytesPerPixel > 1 && c != 1 && c != 3) {
      // split channels
      checkParams = false;

      if (no == 0) {
        initialized[series] = new boolean[initialized[series].length * c];
      }

      for (int i=0; i<c; i++) {
        byte[] b = ImageTools.splitChannels(buf, i, c, bytesPerPixel,
          false, interleaved);

        saveBytes(no * c + i, b, (IFD) ifd.clone(), x, y, w, h);
      }
      checkParams = true;
      return -1;
    }

    formatCompression(ifd);
    byte[][] lut = (cm == null)?null:AWTImageTools.get8BitLookupTable(cm);
    if (lut != null) {
      int[] colorMap = new int[lut.length * lut[0].length];
      for (int i=0; i<lut.length; i++) {
        for (int j=0; j<lut[0].length; j++) {
          colorMap[i * lut[0].length + j] = (int) ((lut[i][j] & 0xff) << 8);
        }
      }
      ifd.putIFDValue(IFD.COLOR_MAP, colorMap);
    }

    int width = retrieve.getPixelsSizeX(series).getValue().intValue();
    int height = retrieve.getPixelsSizeY(series).getValue().intValue();
    ifd.put(new Integer(IFD.IMAGE_WIDTH), new Long(width));
    ifd.put(new Integer(IFD.IMAGE_LENGTH), new Long(height));

    PositiveFloat px = retrieve.getPixelsPhysicalSizeX(series);
    Double physicalSizeX = px == null ? null : px.getValue();
    if (physicalSizeX == null || physicalSizeX.doubleValue() == 0) {
      physicalSizeX = 0d;
    }
    else physicalSizeX = 1d / physicalSizeX;

    PositiveFloat py = retrieve.getPixelsPhysicalSizeY(series);
    Double physicalSizeY = py == null ? null : py.getValue();
    if (physicalSizeY == null || physicalSizeY.doubleValue() == 0) {
      physicalSizeY = 0d;
    }
    else physicalSizeY = 1d / physicalSizeY;

    ifd.put(IFD.RESOLUTION_UNIT, 3);
    ifd.put(IFD.X_RESOLUTION,
      new TiffRational((long) (physicalSizeX * 1000 * 10000), 1000));
    ifd.put(IFD.Y_RESOLUTION,
      new TiffRational((long) (physicalSizeY * 1000 * 10000), 1000));

    if (!isBigTiff) {
      isBigTiff = (out.length() + 2
          * (width * height * c * bytesPerPixel)) >= 4294967296L;
      if (isBigTiff) {
        throw new FormatException("File is too large; call setBigTiff(true)");
      }
    }

    // write the image
    ifd.put(new Integer(IFD.LITTLE_ENDIAN), new Boolean(littleEndian));
    if (!ifd.containsKey(IFD.REUSE)) {
      ifd.put(IFD.REUSE, out.length());
      out.seek(out.length());
    }
    else {
      out.seek((Long) ifd.get(IFD.REUSE));
    }
    
    ifd.putIFDValue(IFD.PLANAR_CONFIGURATION,
      interleaved || getSamplesPerPixel() == 1 ? 1 : 2);

    int sampleFormat = 1;
    if (FormatTools.isSigned(type)) sampleFormat = 2;
    if (FormatTools.isFloatingPoint(type)) sampleFormat = 3;
    ifd.putIFDValue(IFD.SAMPLE_FORMAT, sampleFormat);

    int index = no;
    int realSeries = getSeries();
    for (int i=0; i<realSeries; i++) {
      setSeries(i);
      index += getPlaneCount();
    }
    setSeries(realSeries);
    return index;
  }

  // -- FormatWriter API methods --

  /* (non-Javadoc)
   * @see loci.formats.FormatWriter#close()
   */
  @Override
  public void close() throws IOException {
    super.close();
    if (in != null) {
      in.close();
    }
  }

  /* @see loci.formats.FormatWriter#getPlaneCount() */
  public int getPlaneCount() {
    MetadataRetrieve retrieve = getMetadataRetrieve();
    int c = getSamplesPerPixel();
    int type = FormatTools.pixelTypeFromString(
      retrieve.getPixelsType(series).toString());
    int bytesPerPixel = FormatTools.getBytesPerPixel(type);

    if (bytesPerPixel > 1 && c != 1 && c != 3) {
      return super.getPlaneCount() * c;
    }
    return super.getPlaneCount();
  }

  // -- IFormatWriter API methods --

  /**
   * @see loci.formats.IFormatWriter#saveBytes(int, byte[], int, int, int, int)
   */
  public void saveBytes(int no, byte[] buf, int x, int y, int w, int h)
    throws FormatException, IOException
  {
    IFD ifd = new IFD();
    if (!sequential) {
      TiffParser parser = new TiffParser(currentId);
      try {
        long[] ifdOffsets = parser.getIFDOffsets();
        if (no < ifdOffsets.length) {
          ifd = parser.getIFD(ifdOffsets[no]);
        }
      }
      finally {
        RandomAccessInputStream tiffParserStream = parser.getStream();
        if (tiffParserStream != null) {
          tiffParserStream.close();
        }
      }
    }

    saveBytes(no, buf, ifd, x, y, w, h);
  }

  /* @see loci.formats.IFormatWriter#canDoStacks(String) */
  public boolean canDoStacks() { return true; }

  /* @see loci.formats.IFormatWriter#getPixelTypes(String) */
  public int[] getPixelTypes(String codec) {
    if (codec != null && codec.equals(COMPRESSION_JPEG)) {
      return new int[] {FormatTools.INT8, FormatTools.UINT8,
        FormatTools.INT16, FormatTools.UINT16};
    }
    else if (codec != null && codec.equals(COMPRESSION_J2K)) {
      return new int[] {FormatTools.INT8, FormatTools.UINT8,
        FormatTools.INT16, FormatTools.UINT16, FormatTools.INT32,
        FormatTools.UINT32, FormatTools.FLOAT};
    }
    return new int[] {FormatTools.INT8, FormatTools.UINT8, FormatTools.INT16,
      FormatTools.UINT16, FormatTools.INT32, FormatTools.UINT32,
      FormatTools.FLOAT, FormatTools.DOUBLE};
  }

  // -- TiffWriter API methods --

  /**
   * Sets whether or not BigTIFF files should be written.
   * This flag is not reset when close() is called.
   */
  public void setBigTiff(boolean bigTiff) {
    FormatTools.assertId(currentId, false, 1);
    isBigTiff = bigTiff;
  }

  // -- Helper methods --

  private void setupTiffSaver() throws IOException {
    out.close();
    out = new RandomAccessOutputStream(currentId);
    tiffSaver = new TiffSaver(out, currentId);

    MetadataRetrieve retrieve = getMetadataRetrieve();
    Boolean bigEndian = retrieve.getPixelsBinDataBigEndian(series, 0);
    boolean littleEndian = bigEndian == null ?
      false : !bigEndian.booleanValue();

    tiffSaver.setWritingSequentially(sequential);
    tiffSaver.setLittleEndian(littleEndian);
    tiffSaver.setBigTiff(isBigTiff);
    tiffSaver.setCodecOptions(options);
  }

}
