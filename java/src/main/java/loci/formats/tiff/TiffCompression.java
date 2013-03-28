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
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import loci.common.DataTools;
import loci.common.enumeration.CodedEnum;
import loci.common.enumeration.EnumException;
import loci.formats.FormatException;
import loci.formats.UnsupportedCompressionException;
import loci.formats.codec.Codec;
import loci.formats.codec.CodecOptions;
import loci.formats.codec.JPEG2000Codec;
import loci.formats.codec.JPEG2000CodecOptions;
import loci.formats.codec.JPEGCodec;
import loci.formats.codec.LZWCodec;
import loci.formats.codec.LuraWaveCodec;
import loci.formats.codec.NikonCodec;
import loci.formats.codec.PackbitsCodec;
import loci.formats.codec.PassthroughCodec;
import loci.formats.codec.ZlibCodec;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class for performing compression operations with a TIFF file.
 *
 * <dl><dt><b>Source code:</b></dt>
 * <dd><a href="http://trac.openmicroscopy.org.uk/ome/browser/bioformats.git/components/bio-formats/src/loci/formats/tiff/TiffCompression.java">Trac</a>,
 * <a href="http://git.openmicroscopy.org/?p=bioformats.git;a=blob;f=components/bio-formats/src/loci/formats/tiff/TiffCompression.java;hb=HEAD">Gitweb</a></dd></dl>
 *
 * @author Curtis Rueden ctrueden at wisc.edu
 * @author Eric Kjellman egkjellman at wisc.edu
 * @author Melissa Linkert melissa at glencoesoftware.com
 * @author Chris Allan callan at blackcat.ca
 */
public enum TiffCompression implements CodedEnum {

  // (TIFF code, codec, codec name)
  DEFAULT_UNCOMPRESSED(0, new PassthroughCodec(), "Uncompressed"),
  UNCOMPRESSED(1, new PassthroughCodec(), "Uncompressed"),
  CCITT_1D(2, null, "CCITT Group 3 1-Dimensional Modified Huffman"),
  GROUP_3_FAX(3, null, "CCITT T.4 bi-level encoding (Group 3 Fax)"),
  GROUP_4_FAX(4, null, "CCITT T.6 bi-level encoding (Group 4 Fax)"),
  LZW(5, new LZWCodec(), "LZW"),
  OLD_JPEG(6, new JPEGCodec(), "Old JPEG"),
  JPEG(7, new JPEGCodec(), "JPEG"),
  PACK_BITS(32773, new PackbitsCodec(), "PackBits"),
  PROPRIETARY_DEFLATE(32946, new ZlibCodec(), "Deflate (Zlib)"),
  DEFLATE(8, new ZlibCodec(), "Deflate (Zlib)"),
  THUNDERSCAN(32809, null, "Thunderscan"),
  JPEG_2000(33003, new JPEG2000Codec(), "JPEG-2000") {
    @Override
    public CodecOptions getCompressionCodecOptions(IFD ifd)
        throws FormatException {
      return getCompressionCodecOptions(ifd, null);
    }

    @Override
    public CodecOptions getCompressionCodecOptions(IFD ifd, CodecOptions opt)
    throws FormatException {
      CodecOptions options = super.getCompressionCodecOptions(ifd, opt);
      options.lossless = true;
      JPEG2000CodecOptions j2k = JPEG2000CodecOptions.getDefaultOptions(options);
      if (opt instanceof JPEG2000CodecOptions) {
        JPEG2000CodecOptions o = (JPEG2000CodecOptions) opt;
        j2k.numDecompositionLevels = o.numDecompositionLevels;
        j2k.resolution = o.resolution;
        if (o.codeBlockSize != null)
          j2k.codeBlockSize = o.codeBlockSize;
        if (o.quality > 0)
          j2k.quality = o.quality;
      }
      return j2k;
    }
  },
  JPEG_2000_LOSSY(33004, new JPEG2000Codec(), "JPEG-2000 Lossy") {
    @Override
    public CodecOptions getCompressionCodecOptions(IFD ifd)
        throws FormatException {
      return getCompressionCodecOptions(ifd, null);
    }
    
    @Override
    public CodecOptions getCompressionCodecOptions(IFD ifd, CodecOptions opt)
    throws FormatException {
      CodecOptions options = super.getCompressionCodecOptions(ifd, opt);
      options.lossless = false;
      JPEG2000CodecOptions j2k = JPEG2000CodecOptions.getDefaultOptions(options);
      if (opt instanceof JPEG2000CodecOptions) {
        JPEG2000CodecOptions o = (JPEG2000CodecOptions) opt;
        j2k.numDecompositionLevels = o.numDecompositionLevels;
        j2k.resolution = o.resolution;
        if (o.codeBlockSize != null)
          j2k.codeBlockSize = o.codeBlockSize;
        if (o.quality > 0)
          j2k.quality = o.quality;
      }
      return j2k;
    }
  },
  ALT_JPEG2000(33005, new JPEG2000Codec(), "JPEG-2000") {
    @Override
    public CodecOptions getCompressionCodecOptions(IFD ifd)
        throws FormatException
    {
      return getCompressionCodecOptions(ifd, null);
    }
    
    @Override
    public CodecOptions getCompressionCodecOptions(IFD ifd, CodecOptions opt)
    throws FormatException {
      CodecOptions options = super.getCompressionCodecOptions(ifd, opt);
      options.lossless = true;
      JPEG2000CodecOptions j2k = JPEG2000CodecOptions.getDefaultOptions(options);
      if (opt instanceof JPEG2000CodecOptions) {
        JPEG2000CodecOptions o = (JPEG2000CodecOptions) opt;
        j2k.numDecompositionLevels = o.numDecompositionLevels;
        j2k.resolution = o.resolution;
        if (o.codeBlockSize != null)
          j2k.codeBlockSize = o.codeBlockSize;
        if (o.quality > 0)
          j2k.quality = o.quality;
      }
      return j2k;
    }
  },
  ALT_JPEG(33007, new JPEGCodec(), "JPEG"),
  OLYMPUS_JPEG2000(34712, new JPEG2000Codec(), "JPEG-2000") {
    @Override
    public CodecOptions getCompressionCodecOptions(IFD ifd)
        throws FormatException
    {
      return getCompressionCodecOptions(ifd, null);
    }
    
    @Override
    public CodecOptions getCompressionCodecOptions(IFD ifd, CodecOptions opt)
    throws FormatException {
      CodecOptions options = super.getCompressionCodecOptions(ifd, opt);
      options.lossless = true;
      JPEG2000CodecOptions j2k = JPEG2000CodecOptions.getDefaultOptions(options);
      if (opt instanceof JPEG2000CodecOptions) {
        JPEG2000CodecOptions o = (JPEG2000CodecOptions) opt;
        j2k.numDecompositionLevels = o.numDecompositionLevels;
        j2k.resolution = o.resolution;
        if (o.codeBlockSize != null)
          j2k.codeBlockSize = o.codeBlockSize;
        if (o.quality > 0)
          j2k.quality = o.quality;
      }
      return j2k;
    }
 
  },
  NIKON(34713, new NikonCodec(), "Nikon"),
  LURAWAVE(65535, new LuraWaveCodec(), "LuraWave");

  // -- Constants --

  private static final Logger LOGGER =
    LoggerFactory.getLogger(TiffCompression.class);

  /** Code for the TIFF compression in the actual TIFF file. */
  private int code;

  /** TIFF compression codec. */
  private Codec codec;

  /** Name of the TIFF compression codec. */
  private String codecName;

  /** Reverse lookup of code to TIFF compression enumerate value. */
  private static final Map<Integer, TiffCompression> lookup =
    getCompressionMap();

  private static Map<Integer, TiffCompression> getCompressionMap() {
    Map<Integer, TiffCompression> lookup =
      new HashMap<Integer, TiffCompression>();
    for (TiffCompression v : EnumSet.allOf(TiffCompression.class)) {
      lookup.put(v.getCode(), v);
    }
    return lookup;
  }

  // -- TiffCompression methods --

  /**
   * Default constructor.
   * @param code Integer "code" for the TIFF compression type.
   * @param codec TIFF compression codec.
   * @param codecName String name of the compression type.
   */
  private TiffCompression(int code, Codec codec, String codecName) {
    this.code = code;
    this.codec = codec;
    this.codecName = codecName;
  }

  /**
   * Retrieves a TIFF compression instance by code.
   * @param code Integer "code" for the TIFF compression type.
   * @return See above.
   */
  public static TiffCompression get(int code) {
    TiffCompression toReturn = lookup.get(code);
    if (toReturn == null) {
      throw new EnumException(
          "Unable to find TiffCompresssion with code: " + code);
    }
    return toReturn;
  }

  /* (non-Javadoc)
   * @see loci.common.CodedEnum#getCode()
   */
  public int getCode() {
    return code;
  }

  /**
   * Retrieves the name of the TIFF compression codec.
   * @return See above.
   */
  public String getCodecName() {
    return codecName;
  }

  // -- TiffCompression methods - decompression --

  /** Decodes a strip of data. */
  public byte[] decompress(byte[] input, CodecOptions options)
    throws FormatException, IOException
  {
    if (codec == null) {
      throw new UnsupportedCompressionException(
          "Sorry, " + getCodecName() + " compression mode is not supported");
    }
    return codec.decompress(input, options);
  }

  /** Undoes in-place differencing according to the given predictor value. */
  public static void undifference(byte[] input, IFD ifd)
    throws FormatException
  {
    int predictor = ifd.getIFDIntValue(IFD.PREDICTOR, 1);
    if (predictor == 2) {
      LOGGER.debug("reversing horizontal differencing");
      int[] bitsPerSample = ifd.getBitsPerSample();
      int len = bitsPerSample.length;
      long width = ifd.getImageWidth();
      boolean little = ifd.isLittleEndian();
      int planarConfig = ifd.getPlanarConfiguration();

      int bytes = ifd.getBytesPerSample()[0];

      if (planarConfig == 2 || bitsPerSample[len - 1] == 0) len = 1;
      len *= bytes;

      for (int b=0; b<=input.length-bytes; b+=bytes) {
        if (b / len % width == 0) continue;
        int value = DataTools.bytesToInt(input, b, bytes, little);
        value += DataTools.bytesToInt(input, b - len, bytes, little);
        DataTools.unpackBytes(value, input, b, bytes, little);
      }
    }
    else if (predictor != 1) {
      throw new FormatException("Unknown Predictor (" + predictor + ")");
    }
  }

  // -- TiffCompression methods - compression --

  /**
   * Creates a set of codec options for compression.
   * @param ifd The IFD to create codec options for.
   * @return A new codec options instance populated using metadata from
   * <code>ifd</code>.
   */
  public CodecOptions getCompressionCodecOptions(IFD ifd)
    throws FormatException{
    return getCompressionCodecOptions(ifd, null);
  }

  /**
   * Creates a set of codec options for compression.
   * @param ifd The IFD to create codec options for.
   * @return A new codec options instance populated using metadata from
   * <code>ifd</code>.
   * @param opt The codec options to copy.
   */
  public CodecOptions getCompressionCodecOptions(IFD ifd, CodecOptions opt)
    throws FormatException{
    if (ifd == null)
      throw new IllegalArgumentException("No IFD specified.");
    if (opt == null) opt = CodecOptions.getDefaultOptions();
    CodecOptions options = new CodecOptions(opt);
    options.width = (int) ifd.getImageWidth();
    options.height = (int) ifd.getImageLength();
    options.bitsPerSample = ifd.getBitsPerSample()[0];
    options.channels = ifd.getSamplesPerPixel();
    options.littleEndian = ifd.isLittleEndian();
    options.interleaved = true;
    options.signed = false;
    return options;
  }
  
  /** Encodes a strip of data. */
  public byte[] compress(byte[] input, CodecOptions options)
    throws FormatException, IOException
  {
    if (codec == null) {
      throw new FormatException(
          "Sorry, " + getCodecName() + " compression mode is not supported");
    }
    return codec.compress(input, options);
  }

  /** Performs in-place differencing according to the given predictor value. */
  public static void difference(byte[] input, IFD ifd) throws FormatException {
    int predictor = ifd.getIFDIntValue(IFD.PREDICTOR, 1);
    if (predictor == 2) {
      LOGGER.debug("performing horizontal differencing");
      int[] bitsPerSample = ifd.getBitsPerSample();
      long width = ifd.getImageWidth();
      boolean little = ifd.isLittleEndian();
      int planarConfig = ifd.getPlanarConfiguration();
      int bytes = ifd.getBytesPerSample()[0];
      int len = bytes * (planarConfig == 2 ? 1 : bitsPerSample.length);

      for (int b=input.length-bytes; b>=0; b-=bytes) {
        if (b / len % width == 0) continue;
        int value = DataTools.bytesToInt(input, b, bytes, little);
        value -= DataTools.bytesToInt(input, b - len, bytes, little);
        DataTools.unpackBytes(value, input, b, bytes, little);
      }
    }
    else if (predictor != 1) {
      throw new FormatException("Unknown Predictor (" + predictor + ")");
    }
  }

}
