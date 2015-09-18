/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */
package org.cellprofiler.runnablequeue;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

public class TestRunnableQueue {
	Thread t;
	@Before
	public void setUp() {
		RunnableQueue rq = new RunnableQueue();
		t = new Thread(rq);
		t.setContextClassLoader(null);
		t.start();
	}
	
	@After
	public void tearDown() {
		try {
			RunnableQueue.stop();
			t.join();
		} catch (InterruptedException e) {
			throw new AssertionError("Thread unexpectedly interrupted during join");
		} catch (ExecutionException e) {
			throw new AssertionError("Stop's runnable unexpectedly threw an exception.");
		}
	}
	
	@Test
	public void testStartStop() {
	}

	@Test
	public void testEnqueue() {
		final String [] putSomethingHere = new String [] { null };
		final String something = "Something";
		
		Runnable myRunnable = new Runnable() {
			public void run() {
				synchronized(this) {
					putSomethingHere[0] = something;
					this.notify();
				}
			}
		};
		try {
			RunnableQueue.enqueue(myRunnable);
			synchronized(myRunnable) {
				while (putSomethingHere[0] == null) {
					myRunnable.wait();
				}
			}
		} catch (InterruptedException e) {
			throw new AssertionError("Thread unexpectedly interrupted during enqueue");
		}
		assertEquals(putSomethingHere[0], something);
	}
	@Test
	public void testExecute() {
		final String [] putSomethingHere = new String [] { null };
		final String something = "Something";
		
		Runnable myRunnable = new Runnable() {
			public void run() {
				putSomethingHere[0] = something;
			}
		};
		try {
			RunnableQueue.execute(myRunnable);
		} catch (InterruptedException e) {
			throw new AssertionError("Thread unexpectedly interrupted during enqueue");
		} catch (ExecutionException e) {
			throw new AssertionError("Runnable unexpectedly threw an exception");
		}
		assertEquals(putSomethingHere[0], something);
	}
	@Test
	public void testExecuteV() {
		final String something = "Something";
		Callable<String> myCallable = new Callable<String>() {

			public String call() throws Exception {
				return something;
			}
		};
		try {
			assertEquals(RunnableQueue.execute(myCallable), something);
		} catch (InterruptedException e) {
			throw new AssertionError("Thread unexpectedly interrupted during enqueue");
		} catch (ExecutionException e) {
			throw new AssertionError("Runnable unexpectedly threw an exception");
		}
	}
	@Test
	public void testContextClassLoader() {
		Callable<ClassLoader> myCallable = new Callable<ClassLoader> () {
			public ClassLoader call() throws Exception {
				return Thread.currentThread().getContextClassLoader();
			}
		};
		try {
			assertNotNull(RunnableQueue.execute(myCallable));
		} catch (InterruptedException e) {
			fail();
		} catch (ExecutionException e) {
			fail();
		}
	}
	@Test
	public void testExceptionProof(){
		Runnable myRunnable = new Runnable() {
			public void run() {
				int [] a = new int [] { 1, 2, 3 };
				@SuppressWarnings("unused")
				int b = a[a.length];
			}
		};
		try {
			RunnableQueue.enqueue(myRunnable);
		} catch (InterruptedException e) {
			throw new AssertionError("Thread unexpectedly interrupted during enqueue");
		}
		testEnqueue();
	}
}
