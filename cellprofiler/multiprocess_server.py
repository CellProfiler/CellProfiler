"""
Designed to act as a server for utilizing multiple cores on a machine and 
processing in parallel
"""

import multiprocessing
import StringIO
import zlib
import tempfile
import os
import logging

from cellprofiler.distributed import JobInfo,fetch_work,have_nuageux
from cellprofiler.modules.mergeoutputfiles import MergeOutputFiles
from cellprofiler.pipeline import Pipeline
import cellprofiler.preferences as cpprefs

# whether CP should run multiprocessing (changed by preferences, or by command line)
force_run_multiprocess = False
def run_multiprocess():
    return (force_run_multiprocess or cpprefs.get_run_multiprocess())

def worker_looper(url,job_nums,lock):
    has_work = True
    while has_work:
        with lock:
            jobinfo = fetch_work(url)
        if(jobinfo):
            if(jobinfo.job_num in job_nums):
                #Assume we have more processes than image sets
                #and have looped around
                break
            job_nums.append(jobinfo.job_num)
            num,code = single_job(jobinfo)                
        else: 
            has_work = False
    
def single_job(jobinfo):

        pipeline = Pipeline()
        try:
            pipeline.load(jobinfo.pipeline_stringio())
            image_set_start = jobinfo.image_set_start
            image_set_end = jobinfo.image_set_end
        except:
            logging.root.error("Can't parse pipeline for distributed work.", exc_info=True)
            return [jobinfo.job_num,'FAILURE']

        measurements = pipeline.run(image_set_start=image_set_start, 
                                    image_set_end=image_set_end,
                                    grouping= None)
        
        #out_measurements = StringIO.StringIO()
        #pipeline.save_measurements(out_measurements, measurements)
        
        jobinfo.report_measurements(pipeline, measurements)
        return [jobinfo.job_num,'SUCCESS']
    
def single_job_local(jobinfo):

        pipeline = Pipeline()
        try:
            pipeline.load(jobinfo.pipeline_stringio())
            image_set_start = jobinfo.image_set_start
            image_set_end = jobinfo.image_set_end
        except:
            logging.root.error("Can't parse pipeline for distributed work.", exc_info=True)
            return [jobinfo.job_num,'FAILURE']

        measurements = pipeline.run(image_set_start=image_set_start, 
                                    image_set_end=image_set_end,
                                    grouping= None)
        
        out_measurements = StringIO.StringIO()
        pipeline.save_measurements(out_measurements, measurements)   
        #jobinfo.report_measurements(pipeline, measurements)
        return out_measurements


def run_multiple_workers(url,num_workers = None):
    if(not num_workers):
        num_workers = multiprocessing.cpu_count()
    
    if(True):
        pool = multiprocessing.Pool(num_workers)
        
        urls = [url for i in range(0,num_workers)]
        
        manager = multiprocessing.Manager()
        #donejobs = manager.Queue()
        jobs = manager.list()
        lock = manager.Lock()

        for url in urls:
            pool.apply_async(worker_looper,args=(url,jobs,lock))
            
        #Note: The results will not be available immediately
        #becaus we haven't joined the pool
        donejobs = jobs
        
    else:
        donejobs = worker_looper(url)
        
    return donejobs

if __name__ == '__main__':
    pass
