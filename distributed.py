import nuageux

work_server = None

# Server
def start_serving(pipeline, port):
    global work_server
    if work_server:
        work_server.stop()
    x = 1
    # CHECK: make sure createbatchfiles is not in the pipeline
    # duplicate pipeline
    # create the image list by calling pipeline.prepare_run 
    # start server, get base URL
    # call prepare_to_create_batch to turn files into URLs
    # compress/encode pipeline
    # compress/encode imagesetlist
    # add jobs for each image
    # start serving
    # if headful, call callback periodically, cancel if requested
    # if headless, we can register a callback for results, and idle until the queue is empty 
    
    # when finished, stop serving


def stop_serving():
    pass
    

def image_server():
    # SECURITY: make sure requested images are in served list
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
