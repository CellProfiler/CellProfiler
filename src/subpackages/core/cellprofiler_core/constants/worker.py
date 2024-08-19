from weakref import WeakSet

import zmq

DEADMAN_START_ADDR = b"inproc://deadmanstart"
DEADMAN_START_MSG = b"STARTED"
NOTIFY_ADDR = b"inproc://notify"
NOTIFY_STOP = b"STOP"
NOTIFY_RUN = b"KeepAlive"
ED_STOP = b"Stop"
ED_CONTINUE = b"Continue"
ED_SKIP = b"Skip"
all_measurements: WeakSet = WeakSet()
