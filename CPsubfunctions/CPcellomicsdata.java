import java.util.zip.InflaterInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.File;

public class CPcellomicsdata {
	public byte [] buffer;
	public int width;
	public int height;
	public int nbits;
	public int nplanes;
	public CPcellomicsdata(String filename) 
	throws java.io.FileNotFoundException, java.io.IOException {
		FileInputStream is = new FileInputStream(new File(filename));
		int cookie = readint(is);
		assert cookie == 16*256*256*256;
		InputStream istream = new InflaterInputStream(is);
		int header_length = readint(istream);
		width = readint(istream);
		height = readint(istream);
		nplanes = readshort(istream);
		nbits = readshort(istream);
		int compression = readint(istream);
		readint(istream);
		int pixel_width = readint(istream);
		int pixel_height = readint(istream);
		int color_used = readint(istream);
		int color_important = readint(istream);
		for (int i=0;i<3;i++) {
			readint(istream);
		}
		buffer = new byte [height*width*nbits/8];
		int offset = 0;
		while(offset < buffer.length){
			int count = istream.read(buffer,offset,buffer.length-offset);
			offset += count;
		}
	}

	private int readint(InputStream is) throws java.io.IOException {
		return is.read() + (is.read()+(is.read() + is.read()*256)*256)*256;
	}
	private int readshort(InputStream is) throws java.io.IOException {
		return is.read() + is.read()*256;
	}
	/**
	 * @param args
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args)  {
		try {
			CPcellomicsdata x = new CPcellomicsdata(args[0]);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

	}

}
