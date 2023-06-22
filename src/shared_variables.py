from multiprocessing import Value, Queue, Lock
from queue import Full

from queue import Queue, Full

class DropOldQueue(Queue):
    def __init__(self, maxsize):
        super().__init__(maxsize)

    def put(self, item, *, block=True, timeout=None):
        # Override put method
        while True:
            try:
                Queue.put(self, item, block=block, timeout=timeout)
                break
            except Full:
                # If queue is full, remove an item and try again
                self.get()


class SharedVariables:
    def __init__(self, queue_limit=1000, ranking_threshold=0.5):
        self._camera_enabled = Value('b', False)
        self._camera_enabled_lock = Lock()
        self._tracking_enabled = Value('b', False)
        self._tracking_enabled_lock = Lock()
        self._ranking_threshold = Value('f', ranking_threshold)
        self._ranking_threshold_lock = Lock()
        self._frame_queue = DropOldQueue(maxsize=queue_limit)
        self._tracking_queue = DropOldQueue(maxsize=queue_limit)
        self._filtered_tracking_queue = DropOldQueue(maxsize=queue_limit)

    @property
    def camera_enabled(self):
        with self._camera_enabled_lock:
            return self._camera_enabled.value

    @camera_enabled.setter
    def camera_enabled(self, value):
        if isinstance(value, bool):
            with self._camera_enabled_lock:
                self._camera_enabled.value = value
        else:
            raise ValueError('camera_enabled must be a boolean')

    @property
    def tracking_enabled(self):
        with self._tracking_enabled_lock:
            return self._tracking_enabled.value

    @tracking_enabled.setter
    def tracking_enabled(self, value):
        if isinstance(value, bool):
            with self._tracking_enabled_lock:
                self._tracking_enabled.value = value
        else:
            raise ValueError('tracking_enabled must be a boolean')

    @property
    def ranking_threshold(self):
        with self._ranking_threshold_lock:
            return self._ranking_threshold.value

    @ranking_threshold.setter
    def ranking_threshold(self, value):
        if isinstance(value, float):
            with self._ranking_threshold_lock:
                self._ranking_threshold.value = value
        else:
            raise ValueError('ranking_threshold must be a float')

    @property
    def frame_queue(self):
        return self._frame_queue

    @property
    def tracking_queue(self):
        return self._tracking_queue

    @property
    def filtered_tracking_queue(self):
        return self._filtered_tracking_queue
