"""
Designed to act as a server for utilizing multiple cores on a machine and 
processing in parallel
"""

import multiprocessing
import StringIO
#import zlib
import tempfile
import os
import logging

from cellprofiler.distributed import JobInfo
from cellprofiler.modules.mergeoutputfiles import MergeOutputFiles
from cellprofiler.pipeline import Pipeline

def run_multi(pipeline,output_file,image_set_start = 1,image_set_end = None,grouping = None):
    """
    Run the pipeline with the provided parameters on as many processes as
    there are cores on the PC. This function runs in the parent process,
    so it is blocking.
    """
    
    #XXX We copied this from distributed.start_server(), may not be necessary
    #XXX Would be nice to be able to handle SaveToSpreadsheet and other modules of similar nature properly
    
    # make sure createbatchfiles is not in the pipeline
    if 'CreateBatchFiles' in [module.module_name for module in pipeline.modules()]:
       # XXX - should offer to ignore?
       raise RuntimeException('CreateBatchFiles should not be used with distributed processing.')
    
    # duplicate pipeline
    pipeline = pipeline.copy()
    
    # create the image list
    image_set_list = pipeline.prepare_run(None, combine_path_and_file=True)
    if not image_set_list:
       raise RuntimeError('Could not create image set list for distributed processing.')
    
    pipeline.prepare_to_create_batch(image_set_list, lambda s: s)
    
    # add a CreateBatchFiles module at the end of the pipeline,
    # and set it up for saving the pipeline state
    module = pipeline.instantiate_module('CreateBatchFiles')
    module.module_num = len(pipeline.modules()) + 1
    pipeline.add_module(module)
    module.wants_default_output_directory.set_value(True)
    module.remote_host_is_windows.set_value(False)
    module.batch_mode.set_value(False)
    #module.distributed_mode.set_value(True)

    # save and compress the pipeline
    pipeline_txt = StringIO.StringIO()
    module.save_pipeline(pipeline, image_set_list, outf=pipeline_txt)
    
    #pipeline_blob = zlib.compress(pipeline_txt.getvalue())
    pipeline_blob = pipeline_txt.getvalue()
    pipeline_fd, pipeline_path = tempfile.mkstemp()
    os.write(pipeline_fd, pipeline_blob)
    os.close(pipeline_fd)

    if(image_set_end == None):
        image_set_end = image_set_list.count()
    if image_set_start == None:
        image_set_start = 1
    
    #We analyze only 1 image_set_list at a time
    
    jobnums = range(image_set_start - 1,image_set_end)
    job_list = []
    completed = {}
    #job_queue = multiprocessing.Queue()
    #fin_queue = multiprocessing.Queue()
    
    for jobnum in jobnums:
        jobinfo = JobInfo("file://%s" % (pipeline_path))
        jobinfo.image_set_start,jobinfo.image_set_end = jobnum + 1,jobnum + 1
        jobinfo.job_num = jobnum
        jobinfo.grouping = grouping
        #job_queue.append(jobinfo)
        job_list.append(jobinfo)
        
    #print 'jobnums: %s' % jobnums
    
    def callback(info_array):
        for info in info_array:
            completed[info[0]] = info[1]
        
    #For testing may want to run sequentially
    if(True): 
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        
        #XXX Believe we can't count on the results to be ordered,
        #but not sure
        pool.map_async(single_job,job_list,callback = callback)
        
        pool.close()
        pool.join()
    else:
        results =  []
        for job in job_list:
            results.append(single_job(job))
        callback(results)

    MergeOutputFiles.merge_files(output_file,
                             [completed[n] for n in completed],
                             force_headless=True)
    
    #Cleanup; delete temp pipeline file 
    os.remove(pipeline_path)
    
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
                                    grouping= jobinfo.grouping)
        
        out_measurements = StringIO.StringIO()
        pipeline.save_measurements(out_measurements, measurements)
        
        #jobinfo.report_measurements(pipeline, measurements)
        return [jobinfo.job_num,out_measurements]
 
if __name__ == '__main__':
    pass
