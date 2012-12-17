/* mac_javabridge_utils.c - Utilities for managing the Java Bridge on OS/X

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2012 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org


See http://docs.oracle.com/javase/6/docs/technotes/guides/jni/spec/jniTOC.html
for JNI documentation.
*/

#include "jni.h"

/**********************************************************
 *
 * MacStartVM
 *
 * Start the VM on its own thread, instantiate the thread's runnable
 * and run it until exit.
 *
 * pVMArgs - a pointer to a JavaVMInitArgs structure
 *           as documented for JNI_CreateJavaVM
 *
 * Returns only after the thread terminates. Exit code other
 * than zero indicates failure.
 **********************************************************/
 
JNIEXPORT int MacStartVM(JavaVM **pVM, JavaVMInitArgs *pVMArgs, const char *class_name);

/**********************************************************
 *
 * EnterJVM - enter the JVM and prevent JVM exit
 *
 * Returns 0 if JVM is running, otherwise -1
 *
 **********************************************************/
JNIEXPORT int EnterJVM();

/**********************************************************
 *
 * ExitJVM() - exit the JVM, decreasing the enter-count
 *
 *
 **********************************************************/
JNIEXPORT void ExitJVM();
