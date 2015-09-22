package ij.plugin;
import ij.*;
import ij.plugin.frame.Editor;
import javax.script.*;

/** Implements the macro editor's Macros/Evaluate JavaScript command 
    on systems running Java 1.6 or later. The JavaScript plugin at
    <http://rsb.info.nih.gov/ij/plugins/download/misc/JavaScript.java>
    is used to evaluate JavaScript on systems running versions
    of Java earlier than 1.6. */
public class JavaScriptEvaluator implements PlugIn, Runnable  {
	private Thread thread;
	private String script;
	private Object result;

	// run script in separate thread
	public void run(String script) {
		if (script.equals("")) return;
		if (!IJ.isJava16())
			{IJ.error("Java 1.6 or later required"); return;}
		this.script = script;
		thread = new Thread(this, "JavaScript"); 
		thread.setPriority(Math.max(thread.getPriority()-2, Thread.MIN_PRIORITY));
		thread.start();
	}

	// run script in current thread
	public String run(String script, String arg) {
		this.script = script;
		run();
		return null;
	}

	public void run() {
		result = null;
		Thread.currentThread().setContextClassLoader(IJ.getClassLoader());
		try {
			ScriptEngineManager scriptEngineManager = new ScriptEngineManager();
			ScriptEngine engine = scriptEngineManager.getEngineByName("ECMAScript");
			if (engine == null)
				{IJ.error("Could not find JavaScript engine"); return;}
			if (engine.getFactory().getEngineName().contains("Nashorn")) {
				engine.eval("load('nashorn:mozilla_compat.js');");
			} else {
				engine.eval("function load(path) {\n"
					+ "  importClass(Packages.sun.org.mozilla.javascript.internal.Context);\n"
					+ "  importClass(Packages.java.io.FileReader);\n"
					+ "  var cx = Context.getCurrentContext();\n"
					+ "  cx.evaluateReader(this, new FileReader(path), path, 1, null);\n"
					+ "}");
			}
			result = engine.eval(script);
		} catch(Throwable e) {
			String msg = e.getMessage();
			if (msg.startsWith("sun.org.mozilla.javascript.internal.EcmaError: "))
				msg = msg.substring(47, msg.length());
			if (msg.startsWith("sun.org.mozilla.javascript.internal.EvaluatorException"))
				msg = "Error"+msg.substring(54, msg.length());
			if (!msg.contains("Macro canceled"))
				IJ.log(msg);
		}
	}
	
	public String toString() {
		return result!=null?""+result:"";
	}

}
