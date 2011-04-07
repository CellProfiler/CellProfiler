import nuageux
import StringIO
import zlib
import cPickle
import hashlib
import time

class Distributor(object):
    def __init__(self)
        self.work_server = None
        self.pipeline = None
        self.image_list = None
        self.URL_map = {}
        self.pipeline_blob = ''

    def start_serving(pipeline, port):
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
        self.image_set_list = pipeline.prepare_run()
        if not self.image_list:
            raise RuntimeException('Could not create image set list for distributed processing.')

        # start server, get base URL
        self.work_server = nuageux.Server('CellProfiler work server', data_callback=self.image_server)
        self.server_URL = self.work_server.base_URL()

        # call prepare_to_create_batch to turn files into URLs
        self.URL_map.clear()
        pipeline.prepare_to_create_batch(self.image_set_list, self.rewrite_to_URL)

        # encode/compress pipeline and modified image_set_list
        img_state = image_set_list.save_state()
        pipeline_txt = StringIO.StringIO()
        pipeline.savetxt(pipeline_txt)
        self.pipeline_blob = zlib.compress(cPickle.dumps((pipeline_txt.getvalue(), img_state)))
        # we use the hash to make sure old results don't pollute new
        # ones, and that workers are fetching what they expect.
        self.pipeline_blob_hash = hashlib.sha1(self.pipeline_blob).hexdigest()

        # add jobs for each image set
        for img_set_index in range(image_set_list.count()):
            self.work_server.add_work("%d %s"%(img_set_index, self.pipeline_blob_hash))
        
        # start serving
        self.work_server.start()

        # if headful, call callback periodically, cancel if requested
        # if headless, we can register a callback for results, and idle until the queue is empty 
        unfinished = img_set_list.count()
        while unfinished > 0:
            time.sleep(5):
            finished_job = self.work_server.fetch_result()
            if finished_job:
                print "finished_job", finished_job
                unfinished -= 1

        # when finished, stop serving
        self.work_server.stop()

    def rewrite_to_URL(self, path, **varargs):
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


    def image_server():
        # SECURITY: make sure requested images are in served list
        # special case: -1 = the pipeline_blob
        pass

    def receive_results():
        # call callback to update progress
        pass

    def image_writer():
        # TO BE DONE
        pass

    # Worker (client)
    def do_work():
        pass
<
