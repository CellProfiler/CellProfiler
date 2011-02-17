// Code adapted from a post at the following URL:
//
// http://markmail.org/message/4kh7yqwbipdgjwxa
//
import java.io.File;

class findlibjvm
{
    public static void main(String args[])
    {
	String java_library_path = System.getProperty("java.library.path");
	String [] paths = java_library_path.split(":");
        for (String path:paths) {
	    File f = new File(path, "libjvm.so");
	    if (f.exists()) {
		System.out.println(path);
                break;
	    }
	}
    }
}
