/* mac_javabridge_utils.c - Utilities for managing the Java Bridge on OS/X
 *
 * CellProfiler is distributed under the GNU General Public License,
 * but this file is licensed under the more permissive BSD license.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2013 Broad Institute
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

JNIEnv *pEnv;

static void *thread_function(void *);

typedef struct {
    JavaVM **pvm;
    const char *class_name;
    JavaVMInitArgs *pVMArgs;
    int result;
    char message[256];
} ThreadFunctionArgs;

pthread_mutex_t start_mutex;
pthread_cond_t start_cv;
int started = 0;
int enter_count = 0;

/**********************************************************
 *
 * EnterJVM - enter the JVM and prevent JVM exit
 *
 * Returns 0 if JVM is running, otherwise -1
 *
 **********************************************************/
int EnterJVM()
{
    int was_started;
    pthread_mutex_lock(&start_mutex);
    was_started = started;
    if (started) {
        enter_count++;
    }
    pthread_mutex_unlock(&start_mutex);
    return was_started? 0:-1;
}

/**********************************************************
 *
 * ExitJVM() - exit the JVM, decreasing the enter-count
 *
 *
 **********************************************************/
void ExitJVM()
{
    pthread_mutex_lock(&start_mutex);
    if (--enter_count == 0) {
        pthread_cond_signal(&start_cv);
    }
    pthread_mutex_unlock(&start_mutex);
}
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
    pthread_t thread;
    pthread_attr_t attr;
    void *retval;
    int result;
    
    threadArgs.pvm = pVM;
    
    pthread_mutex_init(&start_mutex, NULL);
    pthread_cond_init(&start_cv, NULL);
    
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
    signal_start();
    
    klass = (*env)->FindClass(env, pThreadArgs->class_name);
    if ((*env)->ExceptionOccurred(env)) {
        snprintf(pThreadArgs->message, 256, "Failed to find class %s\n",
                 pThreadArgs->class_name);
        pThreadArgs->result = -1;
        goto STOP_VM;
    }
    
    method = (*env)->GetMethodID(env, klass, "<init>", "()V");
    if ((*env)->ExceptionOccurred(env)) {
        (*env)->ExceptionDescribe(env);
        snprintf(pThreadArgs->message, 256, "%s has no default constructor\n",
                 pThreadArgs->class_name);
        pThreadArgs->result = -2;
        goto STOP_VM;
    }
    instance = (*env)->NewObjectA(env, klass, method, NULL);
    if ((*env)->ExceptionOccurred(env)) {
        (*env)->ExceptionDescribe(env);
        snprintf(pThreadArgs->message, 256, "Failed to construct %s\n",
                 pThreadArgs->class_name);
        pThreadArgs->result = -3;
        goto STOP_VM;
    }

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
    pthread_mutex_lock(&start_mutex);
    while (enter_count > 0) {
        pthread_cond_wait(&start_cv, &start_mutex);
    }
    started = 0;
    pthread_mutex_unlock(&start_mutex);
    (*vm)->DestroyJavaVM(vm);
    return NULL;
}
    
    
