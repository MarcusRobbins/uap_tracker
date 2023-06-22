import cv2
from queue import Queue
import threading

class CameraLoop:
    def __init__(self, shared_vars):
        self.shared_vars = shared_vars
        self.cameras = self._detect_cameras()
        self.current_camera = None
        self.frame_queue = Queue()
        
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _detect_cameras(self):
        """Detect available cameras and return their details"""
        index = 0
        cameras = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                cameras.append(index)
            cap.release()
            index += 1
        return cameras


    def get_available_cameras(self):
        return self.cameras

    def _read_frame(self):
        """Read frame from current camera"""
        # This is a simplified example, actual camera reading 
        # would depend on the library and hardware being used
        ret, frame = self.current_camera.read()
        if ret:
            self.frame_queue.put(frame)

    def start_camera(self, camera_id):
        """Start selected camera"""
        # In real application, camera selection might require more complexity
        # Here we assume camera_id corresponds directly to camera index
        self.current_camera = cv2.VideoCapture(camera_id)

    def stop_camera(self):
        """Stop the current camera"""
        if self.current_camera:
            self.current_camera.release()
            self.current_camera = None

    def _process_frames(self):
        """Run camera loop, reading frames and checking status of tracking"""
        while True:
            if self.shared_vars.camera_enabled:
                self._read_frame()
                if self.shared_vars.tracking_enabled:
                    # If tracking is enabled, the frames are processed and added to the queue
                    # The actual frame processing and queueing would depend on your tracking implementation
                    pass
                else:
                    # If tracking is not enabled, the frames might be dropped or handled differently
                    pass
