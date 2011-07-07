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
import socket
import logging
logger = logging.getLogger(__name__)

try:
    import nuageux
    have_nuageux = True
except Exception, e:
    logger.warning("Distributed processing disabled (nuageux library not available).\n",exc_info=True)
    have_nuageux = False

from cellprofiler.pipeline import post_module_runner_done_event
from cellprofiler.modules.mergeoutputfiles import MergeOutputFiles
import cellprofiler.preferences as cpprefs

# whether CP should run distributed (changed by preferences, or by command line)
force_run_distributed = False
def run_distributed():
    return have_nuageux and (force_run_distributed or cpprefs.get_run_distributed())

class Distributor(object):
    def __init__(self, frame=None):
        self.work_server = None
        self.pipeline = None
        self.URL_map = {}
        self.output_file = None
        self.status_callback = None
        self.pipeline_path = None
        self.frame = frame

    def start_serving(self, pipeline, port, output_file, status_callback=None):
        # Make sure the previous server has stopped
        if self.work_server:
            self.work_server.stop()

        self.output_file = output_file
        self.status_callback = status_callback

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

        # start server, get base URL
        self.work_server = nuageux.Server('CellProfiler work server',port= port,data_callback=self.data_server, validate_result=self.validate_result)
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
        module.distributed_mode.set_value(True)

        # save and compress the pipeline
        pipeline_txt = StringIO.StringIO()
        module.save_pipeline(pipeline, image_set_list, outf=pipeline_txt)
        pipeline_blob = zlib.compress(pipeline_txt.getvalue())
        pipeline_fd, pipeline_path = tempfile.mkstemp()
        self.pipeline_path = pipeline_path
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
        self.total_jobs = image_set_list.count()
        self.work_server.start()

    def run_with_yield(self):
        # this function acts like a CP pipeline object, allowing us to
        # use the same code path as a non-distributed computation for
        # tracking results and updating the GUI.

        # Returned results are concatenated into this file, which is
        # passed to MergeOutputFiles in pieces, below.
        finished_fd = tempfile.TemporaryFile(dir=os.path.dirname(self.output_file))
        finished_offsets = []
        while True:
            finished_job = self.work_server.fetch_result()
            if finished_job is not None:
                if finished_job['pipeline_hash'][0] == self.pipeline_blob_hash:
                    data_start = finished_fd.tell()
                    finished_fd.write(finished_job['measurements'][0])
                    finished_offsets += [(data_start, len(finished_job['measurements'][0]))]
                    if self.status_callback:
                        self.status_callback(self.total_jobs, len(finished_offsets))
                else:
                    # out of date result?
                    print "ignored mismatched pipeline hash", finished_job['pipeline_hash'][0], self.pipeline_blob_hash
            else:
                # pretend to be busy
                time.sleep(0.1)

            if len(finished_offsets) == self.total_jobs:
                # when finished, stop serving
                self.stop_serving()
                # merge output files.
                finished_fd.flush()
                def nth_output_file(n):
                    start, numbytes = finished_offsets[n]
                    def create_nth_output():
                        self.__str__
                        tmpfile = tempfile.TemporaryFile(dir=os.path.dirname(self.output_file))
                        finished_fd.seek(start)
                        tmpfile.write(zlib.decompress(finished_fd.read(numbytes)))
                        tmpfile.flush()
                        return tmpfile
                    creator = create_nth_output
                    creator.__str__ = lambda _: "%d-th output file"%(n + 1)
                    return creator
                MergeOutputFiles.merge_files(self.output_file,
                                             [nth_output_file(n) for n in range(self.total_jobs)],
                                             force_headless=True)
                # stop iteration
                return

            # this is part of the pipeline mimicry
            if self.frame:
                post_module_runner_done_event(self.frame)

            # continue to yield None until the work is finished
            yield None

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

    def stop_serving(self):
        self.work_server.stop()
        if self.pipeline_path:
            os.unlink(self.pipeline_path)
            self.pipeline_path = None

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
        self._local = self.base_url[0:4] == 'file'
        self.image_set_start = None
        self.image_set_end = None
        self.pipeline_hash = None
        self.pipeline_blob = None
        self.job_num = None

    def fetch_job(self):
        # fetch the pipeline
        #socket.setdefaulttimeout(15) # python >= 2.6, this can be an argument to urlopen
        #self.pipeline_blob = urllib2.urlopen(self.base_url + '/data/-1').read()
        self._get_pipeline_blob()
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

    def work_done(self):
        return False

    def _get_pipeline_blob(self):
        if(self._local):
            self.pipeline_blob = urllib2.urlopen(self.base_url).read()
        else:
            socket.setdefaulttimeout(15) # python >= 2.6, this can be an argument to urlopen
            self.pipeline_blob = urllib2.urlopen(self.base_url + '/data/-1').read()

    def pipeline_stringio(self):
        if(not self.pipeline_blob):
            self._get_pipeline_blob()
        if(self._local):
            return StringIO.StringIO(self.pipeline_blob)
        else:
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
