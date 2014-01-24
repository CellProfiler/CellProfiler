/* mac_javabridge_utils.c - Utilities for managing the Java Bridge on OS/X
 *
 * CellProfiler is distributed under the GNU General Public License,
 * but this file is licensed under the more permissive BSD license.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2014 Broad Institute
 * All rights reserved.
 *
 * Please see the AUTHORS file for credits.
 *
 * Website: http://www.cellprofiler.org
 *
 * The launching strategy and some of the code are liberally borrowed from
 * the Fiji launcher (ij-launcher/ImageJ.c) (Thanks to Dscho)
 *
 * Copyright 2007-2011 Johannes Schindelin, Mark Longair, Albert Cardona
 * Benjamin Schmid, Erwin Frise and Gregory Jefferis
 *
 * The source is distributed under the BSD license.
*/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "jni.h"
#include <pthread.h>
#include <CoreFoundation/CoreFoundation.h>

JNIEnv *pEnv;

static void *thread_function(void *);

typedef struct {
    JavaVM **pvm;
    const char *class_name;
    JavaVMInitArgs *pVMArgs;
    int result;
    char message[256];
} ThreadFunctionArgs;

/**********************************************************
 *
 * The JVM thread
 *
 **********************************************************/
static pthread_t thread;
 
/**********************************************************
 *
 * JVM start communication
 *
 ***********************************************************/
static pthread_mutex_t start_mutex;
static pthread_cond_t start_cv;
static int started = 0;

/**********************************************************
 *
 * JVM stop communication
 *
 **********************************************************/
static pthread_mutex_t stop_mutex;
static pthread_cond_t stop_cv;
static int stopped = 0;

/**********************************************************
 *
 * Run loop synchronization
 *
 **********************************************************/

#define RLS_BEFORE_START 1
#define RLS_STARTED 2
#define RLS_TERMINATING 3
#define RLS_TERMINATED 4

static pthread_mutex_t run_loop_mutex;
static pthread_cond_t run_loop_cv;
static int run_loop_state = RLS_BEFORE_START;

/**********************************************************
 *
 * MacStartVM
 *
 * Start the VM on its own thread, instantiate the thread's runnable
 * and run it until exit.
 *
 * vm_args - a pointer to a JavaVMInitArgs structure
 *           as documented for JNI_CreateJavaVM
 *
 * Returns only after the thread terminates. Exit code other
 * than zero indicates failure.
 **********************************************************/
 
int MacStartVM(JavaVM **pVM, JavaVMInitArgs *pVMArgs, const char *class_name)
{
    ThreadFunctionArgs threadArgs;
    pthread_attr_t attr;
    void *retval;
    int result;
    
    threadArgs.pvm = pVM;
    
    pthread_mutex_init(&start_mutex, NULL);
    pthread_cond_init(&start_cv, NULL);
    
    pthread_mutex_init(&stop_mutex, NULL);
    pthread_cond_init(&stop_cv, NULL);
    
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    threadArgs.pVMArgs = pVMArgs;
    threadArgs.class_name = class_name;
    threadArgs.result = -1;
 
    /* Start the thread that we will start the JVM on. */
    result = pthread_create(&thread, &attr, thread_function, &threadArgs);
    if (result)
        return result;
    pthread_attr_destroy(&attr);
    pthread_mutex_lock(&start_mutex);
    while (started == 0) {
        pthread_cond_wait(&start_cv, &start_mutex);
    }
    pthread_mutex_unlock(&start_mutex);
    if (threadArgs.result) {
        printf(threadArgs.message);
    }
    return threadArgs.result;
}

/************************************************************
 *
 * Stop the JVM
 *
 ************************************************************/
void MacStopVM()
{
    pthread_mutex_lock(&stop_mutex);
    stopped = 1;
    pthread_cond_signal(&stop_cv);
    pthread_mutex_unlock(&stop_mutex);
    pthread_join(&thread, NULL);
}

static void signal_start()
{
    pthread_mutex_lock(&start_mutex);
    started = 1;
    pthread_cond_signal(&start_cv);
    pthread_mutex_unlock(&start_mutex);
}

static void *thread_function(void *arg)
{
    JNIEnv *env;
    jclass klass;
    jmethodID method;
    jobject instance;
    jthrowable exception;
    JavaVM *vm;
    ThreadFunctionArgs *pThreadArgs = (ThreadFunctionArgs *)arg;
 
    pThreadArgs->result = JNI_CreateJavaVM(&vm, (void **)&env, 
                                           pThreadArgs->pVMArgs);
    *pThreadArgs->pvm = vm;
    if (pThreadArgs->result) {
        strcpy(pThreadArgs->message, "Failed to create Java virtual machine.\n");
        signal_start();
        return NULL;
    }
    
    klass = (*env)->FindClass(env, pThreadArgs->class_name);
    if ((*env)->ExceptionOccurred(env)) {
        snprintf(pThreadArgs->message, 256, "Failed to find class %s\n",
                 pThreadArgs->class_name);
        pThreadArgs->result = -1;
        signal_start();
        goto STOP_VM;
    }
    
    method = (*env)->GetMethodID(env, klass, "<init>", "()V");
    if ((*env)->ExceptionOccurred(env)) {
        (*env)->ExceptionDescribe(env);
        snprintf(pThreadArgs->message, 256, "%s has no default constructor\n",
                 pThreadArgs->class_name);
        pThreadArgs->result = -2;
        signal_start();
        goto STOP_VM;
    }
    instance = (*env)->NewObjectA(env, klass, method, NULL);
    if ((*env)->ExceptionOccurred(env)) {
        (*env)->ExceptionDescribe(env);
        snprintf(pThreadArgs->message, 256, "Failed to construct %s\n",
                 pThreadArgs->class_name);
        pThreadArgs->result = -3;
        signal_start();
        goto STOP_VM;
    }
    signal_start();

    method = (*env)->GetMethodID(env, klass, "run", "()V");
    if ((*env)->ExceptionOccurred(env)) {
        (*env)->ExceptionDescribe(env);
        snprintf(pThreadArgs->message, 256, "%s has no run method\n",
                 pThreadArgs->class_name);
        pThreadArgs->result = -4;
        goto STOP_VM;
    }
    (*env)->CallVoidMethodA(env, instance, method, NULL);
    if ((*env)->ExceptionOccurred(env)) {
        (*env)->ExceptionDescribe(env);
        snprintf(pThreadArgs->message, 256, "Failed to execute run method for %s\n",
                 pThreadArgs->class_name);
        pThreadArgs->result = -5;
        goto STOP_VM;
    }
    
STOP_VM:
    pthread_mutex_lock(&stop_mutex);
    while (stopped == 0) {
        pthread_cond_wait(&stop_cv, &stop_mutex);
    }
    started = 0;
    pthread_mutex_unlock(&stop_mutex);
    (*vm)->DestroyJavaVM(vm);
    return NULL;
}

/**************************************************************************
 *
 * CBPerform - a dummy run loop source context perform callback
 *
 **************************************************************************/
 
static void CBPerform(void *info)
{
}

/*************************************************************************
 *
 * CBObserve - a CFRunLoopObserver callback which is called when the
 *             run loop's state changes
 *
 *************************************************************************/
static void CBObserve(CFRunLoopObserverRef observer, 
                      CFRunLoopActivity activity,
                      void *info)
{
    if (activity == kCFRunLoopEntry) {
        pthread_mutex_lock(&run_loop_mutex);
        if (run_loop_state == RLS_BEFORE_START) {
            run_loop_state = RLS_STARTED;
            pthread_cond_signal(&run_loop_cv);
        }
        pthread_mutex_unlock(&run_loop_mutex);
    }
    if (run_loop_state == RLS_TERMINATING) {
        /* Kill, Kill, Kill */
        CFRunLoopStop(CFRunLoopGetCurrent());
    }
}

/*************************************************************************
 *
 * MacRunLoopInit - Configure the main event loop with an observer and source
 *
 *************************************************************************/
 
void MacRunLoopInit()
{
    CFRunLoopObserverContext observerContext;
    CFRunLoopObserverRef     observerRef;
    CFRunLoopSourceContext   sourceContext;
    CFRunLoopSourceRef       sourceRef;
    
    pthread_mutex_init(&run_loop_mutex, NULL);
    pthread_cond_init(&run_loop_cv, NULL);
    
    memset(&sourceContext, 0, sizeof(sourceContext));
    sourceContext.perform = CBPerform;
    sourceRef = CFRunLoopSourceCreate(kCFAllocatorDefault, 0, &sourceContext);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), sourceRef, kCFRunLoopCommonModes);
}

/*************************************************************************
 *
 * MacRunLoopReset - reset the run loop state to before start
 *
 *************************************************************************/
void MacRunLoopReset()
{
    run_loop_state = RLS_BEFORE_START;
}

/*************************************************************************
 *
 * MacRunLoopRun - run the event loop until stopped
 *
 *************************************************************************/
void MacRunLoopRun()
{
    CFRunLoopRun();
    pthread_mutex_lock(&run_loop_mutex);
    run_loop_state = RLS_TERMINATED;
    pthread_cond_signal(&run_loop_cv);
    pthread_mutex_unlock(&run_loop_mutex);
}

/*************************************************************************
 *
 * MacRunLoopRunInMode - run the event loop until timeout or stopped
 *
 *************************************************************************/
void MacRunLoopRunInMode(double timeInterval)
{
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, timeInterval, 1);
}

/****************************************************************************
 *
 * MacRunLoopStop - stop the Mac run loop
 *
 ****************************************************************************/
 
void MacRunLoopStop()
{
    pthread_mutex_lock(&run_loop_mutex);
    while(1) {
        if (run_loop_state == RLS_BEFORE_START) {
            pthread_cond_wait(&run_loop_cv, &run_loop_mutex);
        } else if (run_loop_state == RLS_STARTED) {
            run_loop_state = RLS_TERMINATING;
            CFRunLoopStop(CFRunLoopGetMain());
            pthread_cond_signal(&run_loop_cv);
            while (run_loop_state == RLS_TERMINATING) {
                pthread_cond_wait(&run_loop_cv, &run_loop_mutex);
            }
            break;
        } else {
            /*
             * Assume either RLS_TERMINATING (called twice) or RLS_TERMINATED
             */
             break;
        }
    }
    pthread_mutex_unlock(&run_loop_mutex);
}

/***************************************************************
 *
 * MacIsMainThread - return true if the run loop of this thread
 *                   is the main run loop
 *
 ***************************************************************/
int MacIsMainThread()
{
    return CFRunLoopGetCurrent() == CFRunLoopGetMain();
}
    
