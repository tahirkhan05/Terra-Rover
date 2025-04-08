import cv2
import threading
import time
from queue import Queue, LifoQueue
from config.settings import settings
from models.object_detection import ObjectDetector
from utils.logger import logger

class StreamProcessor:
    def __init__(self):
        self.object_detector = ObjectDetector(settings.OBJECT_DETECTION_MODEL)
        self.detection_queue = LifoQueue(maxsize=5)  # Only keep latest frames
        self.stop_event = threading.Event()
        self.processing_times = []
        self.max_queue_size = 30
        self.frame_queue = Queue(maxsize=self.max_queue_size)
        self.processing_interval = 0.05  # Target 20 FPS processing
        
        # RTSP optimization parameters
        self.cap_params = {
            cv2.CAP_PROP_BUFFERSIZE: 1,
            cv2.CAP_PROP_FPS: settings.FPS,
            cv2.CAP_PROP_FRAME_WIDTH: settings.FRAME_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: settings.FRAME_HEIGHT,
            cv2.CAP_PROP_HW_ACCELERATION: cv2.VIDEO_ACCELERATION_ANY
        }

    def capture_frames(self):
        """Optimized RTSP capture with Windows-specific settings"""
        cap = cv2.VideoCapture(settings.RTSP_URL)
        
        # Set optimized parameters
        for prop, value in self.cap_params.items():
            cap.set(prop, value)
        
        # Add connection timeout and retry logic
        retries = 3
        for attempt in range(retries):
            if cap.isOpened():
                break
            logger.warning(f"RTSP connection attempt {attempt + 1} failed")
            time.sleep(2)
        
        if not cap.isOpened():
            logger.error(f"RTSP stream failed to open after {retries} attempts")
            self.stop_event.set()
            return

        logger.info("RTSP capture started")
        while not self.stop_event.is_set():
            try:
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame capture failed - retrying")
                    time.sleep(0.1)
                    continue

                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)
                
                elapsed = time.time() - start
                sleep_time = max(0, (1/settings.FPS) - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Capture error: {str(e)}")
                time.sleep(0.5)
        
        cap.release()
        logger.info("RTSP capture stopped")

    def process_detections(self):
        """Optimized detection pipeline"""
        while not self.stop_event.is_set():
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    start = time.time()
                    
                    # Skip processing if queue is backing up
                    if self.frame_queue.qsize() > self.max_queue_size * 0.8:
                        continue
                        
                    detections, annotated_frame = self.object_detector.detect_objects(frame)
                    
                    # Store metrics
                    proc_time = time.time() - start
                    if proc_time < 1.0:  # Ignore outliers
                        self.processing_times.append(proc_time)
                        if len(self.processing_times) > 100:
                            self.processing_times.pop(0)
                    
                    # Update detection queue
                    if self.detection_queue.full():
                        self.detection_queue.get_nowait()
                    self.detection_queue.put((detections, annotated_frame))
                    
                    # Adaptive sleep to maintain processing rate
                    elapsed = time.time() - start
                    sleep_time = max(0, self.processing_interval - elapsed)
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Detection error: {str(e)}")
                time.sleep(0.01)
    
    def start(self):
        capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        process_thread = threading.Thread(target=self.process_detections, daemon=True)
        
        capture_thread.start()
        process_thread.start()
        
        return capture_thread, process_thread
    
    def stop(self):
        self.stop_event.set()
    
    def get_latest_detection(self):
        if not self.detection_queue.empty():
            return self.detection_queue.get()
        return None, None