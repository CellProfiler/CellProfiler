import ij.ImagePlus;
import ij.ImageJ;
import ij.macro.*;
import ij.process.*;
import ij.Menus;
import ij.IJ;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.*;
import ij.WindowManager;
import ij.gui.ImageWindow;
import java.lang.StringBuffer;
import java.lang.String;
import java.net.*;

/*
   This is the client program for running ImageJ in a separate process from
   CellProfiler.
   
   Usage: java TCPClient port#
*/

class TCPClient {    
    public static final byte[] intToByteArray(int value) {
        return new byte[] {
                (byte)(value >>> 24),
                (byte)(value >>> 16),
                (byte)(value >>> 8),
                (byte)value};
    }

    public static byte[] floatArrayToByteArray(float pixels[]){    
        byte bytes[] = new byte[pixels.length * 4];
        ByteBuffer bb = ByteBuffer.wrap(bytes);
        FloatBuffer fb = bb.asFloatBuffer();
        fb.put(pixels);
        return bytes;
    }
    
    public static void main(String argv[]) throws Exception{
        String output;
        assert argv.length == 1;
        System.out.println("Socket to "+argv[0]);
        Socket clientSocket = new Socket("localhost", Integer.valueOf(argv[0]).intValue());

        DataOutputStream to_server = new DataOutputStream(clientSocket.getOutputStream());
        DataInputStream from_server = new DataInputStream(clientSocket.getInputStream());

        InterProcessIJBridge ijb = new InterProcessIJBridge();
    
        int result;
        while(true) {
            System.out.println("<CLIENT> reading from socket...");
            int msg_size = from_server.readInt();
            System.out.println("<CLIENT> message size: "+msg_size);
            
            byte[] cmd = new byte[8];
            result = from_server.read(cmd, 0, 8);
            String inp = new String(cmd);
            System.out.println("<CLIENT> command: " + inp);
            
            byte[] bites = new byte[msg_size];
            from_server.readFully(bites, 0, msg_size);
        
            System.out.println("<CLIENT> got " + inp);
            
            if (inp.startsWith("quit")){
                to_server.write(intToByteArray(0));
                to_server.writeBytes("success ");
                to_server.flush();
                clientSocket.close();
                ijb.quit_imagej();
                break;
            } 
            else if (inp.startsWith("inject")){
                ijb.inject_image(bites);
                to_server.write(intToByteArray(0));
                to_server.writeBytes("success ");
                to_server.flush();
            }
            else if (inp.startsWith("getimg")){
                float[] pixels = ijb.get_current_image();
                byte[] bytes = floatArrayToByteArray(pixels);
                // data size
                to_server.write(intToByteArray(bytes.length + 8));
                // message
                to_server.writeBytes("success ");
                // data
                to_server.write(intToByteArray(ijb.get_current_image_width()));
                to_server.write(intToByteArray(ijb.get_current_image_height()));
                to_server.write(bytes);
                to_server.flush();
            }
            else if (inp.startsWith("macro")){ 
                ijb.execute_macro(new String(bites));
                to_server.write(intToByteArray(0));
                to_server.writeBytes("success ");
                to_server.flush();
            }
            else if (inp.startsWith("command")){ 
                // TODO: need to handle command "options"
                ijb.execute_command(new String(bites));
                to_server.write(intToByteArray(0));
                to_server.writeBytes("success ");
                to_server.flush();
            }
            else if (inp.startsWith("getcmds")){ 
                Enumeration cmds = ijb.get_commands();
                
                String cmd_list = "";
                while (cmds.hasMoreElements()) {
                    String ijcmd = (String)cmds.nextElement();
                    if (cmd_list.length() == 0) {
                        cmd_list = cmd_list + ijcmd;
                    } else {
                        cmd_list = cmd_list + "\n" + ijcmd;
                    }
                }
                // data size
                to_server.write(intToByteArray(cmd_list.length()));
                // message
                to_server.writeBytes("success ");
                // data
                to_server.writeBytes(cmd_list);
                to_server.flush();
            }
            else if (inp.startsWith("showij")){ 
                ijb.show_imagej();
            }
        }
    }
}


class InterProcessIJBridge {
    public ImageJ ij;

    public InterProcessIJBridge() {
        ij = new ImageJ();
    }

    public void inject_image(byte[] data) {        
        System.out.println("<CLIENT> injecting");
        ByteArrayInputStream bis = new ByteArrayInputStream(data);
        DataInputStream in = new DataInputStream(bis);
        try {
            int w = 0;
            int h = 0;
            w = in.readInt();
            h = in.readInt();
            byte[] bytes = new byte[w * h * 4]; // 4 bytes per pixel
            in.readFully(bytes, 0, w * h *4);
            in.close();

            ByteBuffer bb = ByteBuffer.wrap(bytes);
            float[] pixels = new float[w * h];
            for(int i=0; i<w*h; i+=1){
                pixels[i] = bb.getFloat();
            }
            
            FloatProcessor fp = new FloatProcessor(w, h, pixels, null);
            ImagePlus imageplus = new ImagePlus("", fp);
            ImageWindow im_window = imageplus.getWindow();
            WindowManager.setCurrentWindow(im_window);
            imageplus.show();
        } catch (java.io.IOException e) {
            System.out.println(e);
            return;
        }
    }

    public int get_current_image_width() throws 
        FileNotFoundException, IOException {
        ImagePlus imageplus = WindowManager.getCurrentImage();
        return imageplus.getWidth();
    }

    public int get_current_image_height() throws 
        FileNotFoundException, IOException {
        ImagePlus imageplus = WindowManager.getCurrentImage();
        return imageplus.getHeight();
    }

    public float[] get_current_image() throws 
        FileNotFoundException, IOException {
        // We currently only handle returning a single channel.
        ImagePlus imageplus = WindowManager.getCurrentImage();
        ImageProcessor ip = imageplus.getProcessor();
        TypeConverter tc = new TypeConverter(ip, false);
        ImageProcessor fp = tc.convertToFloat(null);
        return (float[]) fp.getPixels();
    }

    public void execute_macro(String macro_text) {
        System.err.println("executing macro "+macro_text);
        show_imagej();
        Interpreter interp = new Interpreter();
        interp.run(macro_text);
    }

    public void show_imagej() {
        ij.setVisible(true);
        ij.toFront();
    }

    public Enumeration get_commands(){
        Hashtable hashtable = Menus.getCommands();
        if (hashtable == null){
            //
            // This is a little bogus, but works - trick IJ into initializing
            //
            IJ.run("pleaseignorethis");
            hashtable = Menus.getCommands();
            if (hashtable == null){
                return null;
            }
        }
        Enumeration keys = hashtable.keys();
        return keys;
    }

    public void execute_command(String command){
        IJ.run(command);
    }

    public void execute_command(String command, String options){
        IJ.run(command, options);        
    }

    public void quit_imagej(){
        System.out.println("<CLIENT> exiting");
        java.lang.System.exit(0);
        // prompts to save if images are open which we don't want
        // ij.quit();
    }

}

