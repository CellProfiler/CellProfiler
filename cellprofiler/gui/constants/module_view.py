import threading
from queue import Queue

import wx

WARNING_COLOR = wx.Colour(224, 224, 0, 255)
RANGE_TEXT_WIDTH = 40  # number of pixels in a range text box TO_DO - calculate it
ABSOLUTE = "Absolute"
FROM_EDGE = "From edge"
CHECK_TIMEOUT_SEC = 10
EDIT_TIMEOUT_SEC = 5
PRI_VALIDATE_DISPLAY = 0
PRI_VALIDATE_BACKGROUND = 1
validation_queue = Queue()
pipeline_queue_thread = None  # global, protected by above lock
request_pipeline_cache = threading.local()  # used to cache the last requested pipeline
validation_queue_keep_running = True
