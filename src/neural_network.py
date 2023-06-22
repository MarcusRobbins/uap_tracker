import threading

class NeuralNetworkProcessing:
    def __init__(self, shared_variables):
        self.shared_vars = shared_variables
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_frames(self):
        while True:
            if self.shared_vars.tracking_enabled and not self.shared_vars.frame_queue.empty():
                frame = self.shared_vars.frame_queue.get()
                track_data = self.get_empty_track_data()
                self.shared_vars.tracking_queue.put(track_data)
                
    @staticmethod
    def get_empty_track_data():
        # Returns empty tracking data.
        return {}
