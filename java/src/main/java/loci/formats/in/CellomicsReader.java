/*
 * #%L
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import loci.common.Location;
import loci.common.RandomAccessInputStream;
import loci.formats.CoreMetadata;
import loci.formats.FormatException;
import loci.formats.FormatReader;
import loci.formats.FormatTools;
import loci.formats.MetadataTools;
import loci.formats.UnsupportedCompressionException;
import loci.formats.codec.ZlibCodec;
import loci.formats.meta.MetadataStore;

import ome.xml.model.enums.NamingConvention;
import ome.xml.model.primitives.NonNegativeInteger;
import ome.xml.model.primitives.PositiveFloat;

/**
 * Reader for Cellomics C01 files.
 *
 * <dl><dt><b>Source code:</b></dt>
 * <dd><a href="http://trac.openmicroscopy.org.uk/ome/browser/bioformats.git/components/bio-formats/src/loci/formats/in/CellomicsReader.java">Trac</a>,
 * <a href="http://git.openmicroscopy.org/?p=bioformats.git;a=blob;f=components/bio-formats/src/loci/formats/in/CellomicsReader.java;hb=HEAD">Gitweb</a></dd></dl>
 */
public class CellomicsReader extends FormatReader {

  // -- Constants --

  public static final int C01_MAGIC_BYTES = 16;

  // -- Fields --

  // A typical Cellomics file name is
  // WHICA-VTI1_090915160001_A01f00o1.DIB
  // The plate name is:
  // WHICA-VTI1_090915160001
  // The well name is A01
  // The site / field is 00
  // the channel is 1
  //
  // The channel prefix can be "o" or "d"
  // Both site and channel are optional.
  //
  // The pattern greedily captures:
  // The plate name in group 1
  // The well name in group 2
  // The field, optionally, in group 3
  // The channel, optionally, in group 4
  private static final Pattern cellomicsPattern = Pattern.compile("(.*)_(\\p{Alpha}\\d{2})(f\\d{2})?([od]\\d+)?[^_]+$");
  private String[] files;

  // -- Constructor --

  /** Constructs a new Cellomics reader. */
  public CellomicsReader() {
    super("Cellomics C01", new String[] {"c01", "dib"});
    domains = new String[] {FormatTools.LM_DOMAIN, FormatTools.HCS_DOMAIN};
    datasetDescription = "One or more .c01 files";
  }

  // -- IFormatReader API methods --

  /* @see loci.formats.IFormatReader#isThisType(RandomAccessInputStream) */
  public boolean isThisType(RandomAccessInputStream stream) throws IOException {
    final int blockLen = 4;
    if (!FormatTools.validStream(stream, blockLen, false)) return false;
    return stream.readInt() == C01_MAGIC_BYTES;
  }

  /* @see loci.formats.IFormatReader#getDomains() */
  public String[] getDomains() {
    FormatTools.assertId(currentId, true, 1);
    return new String[] {FormatTools.HCS_DOMAIN};
  }

  /**
   * @see loci.formats.IFormatReader#openBytes(int, byte[], int, int, int, int)
   */
  public byte[] openBytes(int no, byte[] buf, int x, int y, int w, int h)
    throws FormatException, IOException
  {
    FormatTools.checkPlaneParameters(this, no, buf.length, x, y, w, h);

    int[] zct = getZCTCoords(no);

    String file = files[getSeries() * getSizeC() + zct[1]];
    RandomAccessInputStream s = getDecompressedStream(file);

    int planeSize = FormatTools.getPlaneSize(this);
    s.seek(52 + zct[0] * planeSize);
    readPlane(s, x, y, w, h, buf);
    s.close();

    return buf;
  }

  /* @see loci.formats.IFormatReader#close(boolean) */
  public void close(boolean fileOnly) throws IOException {
    super.close(fileOnly);
    if (!fileOnly) {
      files = null;
    }
  }

  /* @see loci.formats.IFormatReader#getSeriesUsedFiles(boolean) */
  public String[] getSeriesUsedFiles(boolean noPixels) {
    FormatTools.assertId(currentId, true, 1);
    return files;
  }

  /* @see loci.formats.IFormatReader#fileGroupOption(String) */
  public int fileGroupOption(String id) throws FormatException, IOException {
    return FormatTools.MUST_GROUP;
  }

  // -- Internal FormatReader API methods --

  /* @see loci.formats.FormatReader#initFile(String) */
  protected void initFile(String id) throws FormatException, IOException {
    super.initFile(id);

    // look for files with similar names
    Location baseFile = new Location(id).getAbsoluteFile();
    Location parent = baseFile.getParentFile();
    ArrayList<String> pixelFiles = new ArrayList<String>();

    String plateName = getPlateName(baseFile.getName());

    if (plateName != null && isGroupFiles()) {
      String[] list = parent.list();
      for (String f : list) {
        if (plateName.equals(getPlateName(f)) &&
          (checkSuffix(f, "c01") || checkSuffix(f, "dib")))
        {
          Location loc = new Location(parent, f);
          if ((!f.startsWith(".") || !loc.isHidden()) && getChannel(f) >= 0) {
            pixelFiles.add(loc.getAbsolutePath());
          }
        }
      }
    }
    else pixelFiles.add(id);

    files = pixelFiles.toArray(new String[pixelFiles.size()]);
    Arrays.sort(files);

    int wellRows = 0;
    int wellColumns = 0;
    int fields = 0;

    ArrayList<Integer> uniqueRows = new ArrayList<Integer>();
    ArrayList<Integer> uniqueCols = new ArrayList<Integer>();
    ArrayList<Integer> uniqueFields = new ArrayList<Integer>();
    ArrayList<Integer> uniqueChannels = new ArrayList<Integer>();
    for (String f : files) {
      int wellRow = getWellRow(f);
      int wellCol = getWellColumn(f);
      int field = getField(f);
      int channel = getChannel(f);

      if (!uniqueRows.contains(wellRow)) uniqueRows.add(wellRow);
      if (!uniqueCols.contains(wellCol)) uniqueCols.add(wellCol);
      if (!uniqueFields.contains(field)) uniqueFields.add(field);
      if (!uniqueChannels.contains(channel)) uniqueChannels.add(channel);
    }

    fields = uniqueFields.size();
    wellRows = uniqueRows.size();
    wellColumns = uniqueCols.size();

    if (fields * wellRows * wellColumns > files.length) {
      files = new String[] {id};
    }

    final int nEntries = files.length / uniqueChannels.size();
	core = new ArrayList<CoreMetadata>(nEntries);

    for (int i=0; i<nEntries; i++) {
      core.add(new CoreMetadata());
    }

    in = getDecompressedStream(id);

    LOGGER.info("Reading header data");

    in.order(true);
    in.skipBytes(4);

    int x = in.readInt();
    int y = in.readInt();
    int nPlanes = in.readShort();
    int nBits = in.readShort();

    int compression = in.readInt();

    if (x * y * nPlanes * (nBits / 8) + 52 > in.length()) {
      throw new UnsupportedCompressionException(
        "Compressed pixel data is not yet supported.");
    }

    in.skipBytes(4);
    int pixelWidth = 0, pixelHeight = 0;

    if (getMetadataOptions().getMetadataLevel() != MetadataLevel.MINIMUM) {
      pixelWidth = in.readInt();
      pixelHeight = in.readInt();
      int colorUsed = in.readInt();
      int colorImportant = in.readInt();

      LOGGER.info("Populating metadata hashtable");

      addGlobalMeta("Image width", x);
      addGlobalMeta("Image height", y);
      addGlobalMeta("Number of planes", nPlanes);
      addGlobalMeta("Bits per pixel", nBits);
      addGlobalMeta("Compression", compression);
      addGlobalMeta("Pixels per meter (X)", pixelWidth);
      addGlobalMeta("Pixels per meter (Y)", pixelHeight);
      addGlobalMeta("Color used", colorUsed);
      addGlobalMeta("Color important", colorImportant);
    }

    LOGGER.info("Populating core metadata");

    for (int i=0; i<getSeriesCount(); i++) {
      CoreMetadata coreitem = core.get(i);
      coreitem.sizeX = x;
      coreitem.sizeY = y;
      coreitem.sizeZ = nPlanes;
      coreitem.sizeT = 1;
      coreitem.sizeC = uniqueChannels.size();
      coreitem.imageCount = getSizeZ() * getSizeT() * getSizeC();
      coreitem.littleEndian = true;
      coreitem.dimensionOrder = "XYCZT";
      coreitem.pixelType =
        FormatTools.pixelTypeFromBytes(nBits / 8, false, false);
    }

    LOGGER.info("Populating metadata store");

    MetadataStore store = makeFilterMetadata();
    MetadataTools.populatePixels(store, this);

    store.setPlateID(MetadataTools.createLSID("Plate", 0), 0);
    store.setPlateName(plateName, 0);
    store.setPlateRowNamingConvention(NamingConvention.LETTER, 0);
    store.setPlateColumnNamingConvention(NamingConvention.NUMBER, 0);

    int realRows = wellRows;
    int realCols = wellColumns;

    if (files.length == 1) {
      realRows = 1;
      realCols = 1;
    }
    else if (realRows <= 8 && realCols <= 12) {
      realRows = 8;
      realCols = 12;
    }
    else {
      realRows = 16;
      realCols = 24;
    }

    for (int row=0; row<realRows; row++) {
      for (int col=0; col<realCols; col++) {
        int well = row * realCols + col;

        if (files.length == 1) {
          row = getWellRow(files[0]);
          col = getWellColumn(files[0]);
        }

        store.setWellID(MetadataTools.createLSID("Well", 0, well), 0, well);
        store.setWellRow(new NonNegativeInteger(row), 0, well);
        store.setWellColumn(new NonNegativeInteger(col), 0, well);
      }
    }

    for (int i=0; i<getSeriesCount(); i++) {
      String file = files[i * getSizeC()];

      int fieldIndex = getField(file);
      int row = getWellRow(file);
      int col = getWellColumn(file);

      if (files.length == 1) {
        row = 0;
        col = 0;
      }

      String imageID = MetadataTools.createLSID("Image", i);
      store.setImageID(imageID, i);
      if (row < realRows && col < realCols) {

        int wellIndex = row * realCols + col;

        if (files.length == 1) {
          fieldIndex = 0;
        }

        String wellSampleID =
          MetadataTools.createLSID("WellSample", 0, wellIndex, fieldIndex);
        store.setWellSampleID(wellSampleID, 0, wellIndex, fieldIndex);
        store.setWellSampleIndex(
          new NonNegativeInteger(i), 0, wellIndex, fieldIndex);

        store.setWellSampleImageRef(imageID, 0, wellIndex, fieldIndex);
      }
      store.setImageName(
        String.format("Well %s%02d, Field #%02d", 
                      new String(Character.toChars(row+'A')), 
                      col, fieldIndex), i);
    }

    if (getMetadataOptions().getMetadataLevel() != MetadataLevel.MINIMUM) {
      // physical dimensions are stored as pixels per meter - we want them
      // in microns per pixel
      double width = pixelWidth == 0 ? 0.0 : 1000000.0 / pixelWidth;
      double height = pixelHeight == 0 ? 0.0 : 1000000.0 / pixelHeight;

      PositiveFloat sizeX = FormatTools.getPhysicalSizeX(width);
      PositiveFloat sizeY = FormatTools.getPhysicalSizeY(height);
      for (int i=0; i<getSeriesCount(); i++) {
        if (sizeX != null) {
          store.setPixelsPhysicalSizeX(sizeX, 0);
        }
        if (sizeY != null) {
          store.setPixelsPhysicalSizeY(sizeY, 0);
        }
      }
    }
  }

  // -- Helper methods --

  static private Matcher matchFilename(final String filename) {
    final String name = new Location(filename).getName();
    return cellomicsPattern.matcher(name);
  }
  private String getPlateName(final String filename) {
    Matcher m = matchFilename(filename);
    if (m.matches()) {
      return m.group(1);
    }
    return null;
  }

  private String getWellName(String filename) {
    Matcher m = matchFilename(filename);
    if (m.matches()) {
        return m.group(2);
    }
    return null;
  }

  private int getWellRow(String filename) {
    String wellName = getWellName(filename);
    if ((wellName == null) || (wellName.length() < 1) ) return 0;
    int ord = wellName.toUpperCase().charAt(0) - 'A';
    if ((ord < 0) || (ord >= 26)) return 0;
    return ord;
  }

  private int getWellColumn(String filename) {
    String wellName = getWellName(filename);
    if ((wellName == null) || (wellName.length() <= 2)) return 0;
    if (! Character.isDigit(wellName.charAt(1))) return 0;
    if (! Character.isDigit(wellName.charAt(2))) return 0;
    return Integer.parseInt(wellName.substring(1, 3));
  }

  private int getField(String filename) {
    Matcher m = matchFilename(filename);
    if (m.matches() && (m.group(3) != null)) {
      return Integer.parseInt(m.group(3).substring(1));
    }
    return 0;
  }

  private int getChannel(String filename) {
    Matcher m = matchFilename(filename);
    if (m.matches() && (m.group(4) != null)) {
      return Integer.parseInt(m.group(4).substring(1));
    }
    return -1;
  }

  private RandomAccessInputStream getDecompressedStream(String filename)
    throws FormatException, IOException
  {
    RandomAccessInputStream s = new RandomAccessInputStream(filename);
    if (checkSuffix(filename, "c01")) {
      LOGGER.info("Decompressing file");

      s.seek(4);
      ZlibCodec codec = new ZlibCodec();
      byte[] file = codec.decompress(s, null);
      s.close();

      return new RandomAccessInputStream(file);
    }
    return s;
  }

}
