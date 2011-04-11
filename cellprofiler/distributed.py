import os
import os.path
import sys
import StringIO
import zlib
import hashlib
import time
import tempfile
import traceback
import urllib, urllib2

import nuageux
from cellprofiler.modules.mergeoutputfiles import MergeOutputFiles


class Distributor(object):
    def __init__(self):
        self.work_server = None
        self.pipeline = None
        self.URL_map = {}

    def start_serving(self, pipeline, port, output_file):
        # Make sure the previous server has stopped
        if self.work_server:
            self.work_server.stop()

        # make sure createbatchfiles is not in the pipeline
        if 'CreateBatchFiles' in [module.module_name for module in pipeline.modules()]:
            # XXX - should offer to ignore?
            raise RuntimeException('CreateBatchFiles should not be used with distributed processing.')

        # duplicate pipeline
        self.pipeline = pipeline.copy()

        # create the image list
        image_set_list = pipeline.prepare_run(None, combine_path_and_file=True)
        if not image_set_list:
            raise RuntimeError('Could not create image set list for distributed processing.')

        # start server, get base URL
        self.work_server = nuageux.Server('CellProfiler work server', data_callback=self.data_server, validate_result=self.validate_result)
        self.server_URL = self.work_server.base_URL()

        # call prepare_to_create_batch to turn files into URLs
        self.URL_map.clear()
        pipeline.prepare_to_create_batch(image_set_list, self.rewrite_to_URL)

        # add a CreateBatchFiles module at the end of the pipeline,
        # and set it up for saving the pipeline state
        module = pipeline.instantiate_module('CreateBatchFiles')
        module.module_num = len(pipeline.modules()) + 1
        pipeline.add_module(module)
        module.wants_default_output_directory.set_value(True)
        module.remote_host_is_windows.set_value(False)
        module.batch_mode.set_value(False)

        # save and compress the pipeline
        pipeline_txt = StringIO.StringIO()
        module.save_pipeline(pipeline, image_set_list, outf=pipeline_txt)
        pipeline_blob = zlib.compress(pipeline_txt.getvalue())
        pipeline_fd, pipeline_path = tempfile.mkstemp()
        os.write(pipeline_fd, pipeline_blob)
        os.close(pipeline_fd)

        # we use the hash to make sure old results don't pollute new
        # ones, and that workers are fetching what they expect.
        self.pipeline_blob_hash = hashlib.sha1(pipeline_blob).hexdigest()

        # special case for URL_map:  -1 is the pipeline blob
        self.URL_map[-1] = pipeline_path

        # add jobs for each image set
        for img_set_index in range(image_set_list.count()):
            self.work_server.add_work("%d %s"%(img_set_index + 1, self.pipeline_blob_hash))
        
        # start serving
        self.work_server.start()

        # XXX - if headful, call callback periodically, cancel if requested
        # if headless, we can register a callback for results, and idle until the queue is empty 
        unfinished = image_set_list.count()
        finished_fds = []
        while unfinished > 0:
            print "idling - ", unfinished, "jobs remain", self.work_server.base_URL()
            finished_job = self.work_server.fetch_result()
            if finished_job is not None:
                if finished_job['pipeline_hash'][0] != self.pipeline_blob_hash:
                    # out of date result?
                    print "ignored mismatched pipeline hash", finished_job['pipeline_hash'][0], self.pipeline_blob_hash
                    continue
                # store results in a temporary file, in the output directory
                outfd = tempfile.TemporaryFile(dir=os.path.dirname(output_file))
                outfd.write(zlib.decompress(finished_job['measurements'][0]))
                outfd.flush()
                outfd.seek(0)
                finished_fds.append(outfd)
                print "finished image number", finished_job['image_num'][0]
                unfinished -= 
            else:
                time.sleep(1)


        # when finished, stop serving
        self.work_server.stop()
        os.unlink(pipeline_path)

        # merge output files
        MergeOutputFiles.merge_files(output_file, finished_fds, force_headless=True)

    def validate_result(self, result):
        return self.pipeline_blob_hash == result['pipeline_hash'][0]

    def rewrite_to_URL(self, path, **varargs):
        # For now, each image gets an integer, but for debugging,
        # perhaps base64-encoding the path would make debugging
        # easier.
        
        # empty path entries should be ignored
        if path == '':
            return ''

        # XXX - need to do something with regexp_substitution
        if path in self.URL_map:
            img_index = self.URL_map[path]
        else:
            img_index = len(self.URL_map)
            # two way map for validation.  Using strings and ints prevents collisions
            self.URL_map[path] = img_index
            self.URL_map[img_index] = path
        return "%s/data/%s"%(self.server_URL, str(img_index))
            

    def stop_serving():
        pass


    def data_server(self, request):
        try:
            # take just the first element of the request
            req = int(request[0])
            # SECURITY: make sure reqd images are in served list
            return self.URL_map[req]
        except Exception, e:
            print "bad data request", request, e
            return '', 'application/octet-stream'

    def image_writer():
        # TO BE DONE
        pass

class JobInfo(object):
    def __init__(self, base_url):
        self.base_url = base_url
        self.image_set_start = None
        self.image_set_end = None
        self.pipeline_hash = None
        self.pipeline_blob = None
        self.job_num = None

    def fetch_job(self):
        try:
            # fetch the pipeline
            self.pipeline_blob = urllib2.urlopen(self.base_url + '/data/-1').read()
            self.pipeline_hash = hashlib.sha1(self.pipeline_blob).hexdigest()
            # fetch a job
            work_blob = urllib2.urlopen(self.base_url + '/work').read()
            if work_blob == 'NOWORK':
                assert False, "No work to be had..."
            self.job_num, image_num, pipeline_hash = work_blob.split(' ')
            self.image_set_start = int(image_num)
            self.image_set_end = int(image_num)
            print "fetched work:", work_blob
            assert pipeline_hash == self.pipeline_hash, "Mismatched hash, probably out of sync with server"
        except Exception, e:
            sys.stderr.write("Exception fetching work.\n")
            traceback.print_exc()

    def work_done(self):
        return False

    def pipeline_stringio(self):
        return StringIO.StringIO(zlib.decompress(self.pipeline_blob))

    def report_measurements(self, pipeline, measurements):
        out_measurements = StringIO.StringIO()
        pipeline.save_measurements(out_measurements, measurements)
        nuageux.report_result(self.base_url, self.job_num,
                              image_num=str(self.image_set_start),
                              pipeline_hash=self.pipeline_hash,
                              measurements=zlib.compress(out_measurements.getvalue()))

def fetch_work(base_URL):
    jobinfo = JobInfo(base_URL)
    jobinfo.fetch_job()
    return jobinfo
