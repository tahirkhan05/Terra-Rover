import cv2
import time
import json
import os
import threading
import numpy as np
from config.settings import settings
from services.stream_processor import StreamProcessor
from services.speech_processor import SpeechProcessor
from services.image_processor import ImageProcessor
from models.vlm_processor import VLMProcessor
from utils.parallel import ParallelProcessor
from utils.logger import logger

class TerraRover:
    def __init__(self):
        self.stream_processor = StreamProcessor()
        self.speech_processor = SpeechProcessor()
        self.vlm_processor = VLMProcessor()
        self.image_processor = ImageProcessor()
        self.parallel_processor = ParallelProcessor(settings.MAX_WORKERS)
        self.running = False
        self.last_vlm_call = 0
        self.vlm_cooldown = 1.0  # Minimum seconds between VLM calls
        self.processing_voice = False  # Flag to prevent multiple voice queries

    def start(self):
        logger.info("Starting Terra Rover System")
        self.running = True
        
        # Start subsystems
        capture_thread, process_thread = self.stream_processor.start()
        self._start_status_monitor()
        
        # Print startup message with instructions
        print("\n" + "="*50)
        print("Terra Rover System Started")
        print("="*50)
        print("Press 's' to ask a question about what you see")
        print("Press 'q' to quit")
        print("="*50 + "\n")
        
        try:
            while self.running:
                self._process_frame()
                self._handle_input()
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            self._shutdown(capture_thread, process_thread)

    def _process_frame(self):
        """Handle frame processing pipeline"""
        detections, frame = self.stream_processor.get_latest_detection()
        if frame is not None:
            # Save to S3 in parallel
            self.parallel_processor.submit_task(
                self.image_processor.process_and_store_frame,
                frame
            )
            
            # Display
            if not self.image_processor.display_frame(frame):
                self.running = False

    def _handle_input(self):
        """Process user input with cooldown and better feedback"""
        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()
        
        if key == ord('q'):
            self.running = False
        elif key == ord('s') and (current_time - self.last_vlm_call) > self.vlm_cooldown and not self.processing_voice:
            self.last_vlm_call = current_time
            # Directly process voice query in this thread for better debugging
            # Prevents issues with thread execution
            self.processing_voice = True
            try:
                print("\n🔊 Voice query activated!")
                self._process_voice_query()
            finally:
                self.processing_voice = False

    def _process_voice_query(self):
        """Enhanced voice query handling with better debugging"""
        logger.info("Starting voice query processing...")
        
        try:
            # 1. Capture audio - do this synchronously for better feedback
            logger.debug("Starting audio recording...")
            audio = self.speech_processor.record_audio(duration=5)
            if not audio:
                logger.error("No audio data captured")
                print("❌ No audio detected. Please try again.")
                return
                
            # 2. Transcribe
            logger.debug("Starting speech transcription...")
            question = self.speech_processor.transcribe_speech(audio)
            if not question:
                logger.error("No transcription returned")
                return
                
            logger.info(f"Transcribed question: {question}")
            print(f"🎙️ Your question: {question}")
            
            # 3. Get latest frame
            logger.debug("Getting latest frame...")
            detections, frame = self.stream_processor.get_latest_detection()
            if frame is None:
                logger.error("No frame available for processing")
                print("❌ No video frame available to analyze")
                return
            
            # 4. Save frame
            logger.debug("Processing and storing frame...")
            print("🖼️ Processing current frame...")
            s3_path, image_key = self.image_processor.process_and_store_frame(frame)
            if not image_key:
                logger.error("Failed to store frame in S3")
                print("❌ Failed to store image for analysis")
                return
                
            logger.debug(f"Frame stored at: {s3_path}")
            
            # 5. Process with VLM
            logger.debug("Invoking VLM...")
            print("🤖 Analyzing image and generating response...")
            
            # Check if VLM model ID is configured
            if not settings.VLM_MODEL_ID:
                logger.error("VLM model ID not configured")
                print("❌ VLM model not configured in .env file")
                return
                
            response = self.vlm_processor.generate_response(
                query_type='general',
                image_key=image_key,
                question=question
            )
            
            # Pretty print the response
            print("\n" + "="*50)
            print("✅ ANSWER:")
            print(f"{response}")
            print("="*50 + "\n")
            
            logger.info(f"VLM Response: {response}")
            
        except Exception as e:
            logger.error(f"Error in voice query processing: {str(e)}")
            print(f"❌ Error processing your query: {str(e)}")

    def _start_status_monitor(self):
        """Enhanced system monitoring"""
        def monitor():
            while self.running:
                time.sleep(10)  # Reduced frequency
                stats = {
                    "fps": 0,
                    "queue": self.stream_processor.frame_queue.qsize(),
                    "detection_queue": self.stream_processor.detection_queue.qsize(),
                    "processing_time": 0
                }
                
                if self.stream_processor.processing_times:
                    stats["fps"] = 1/np.mean(self.stream_processor.processing_times)
                    stats["processing_time"] = np.mean(self.stream_processor.processing_times)
                
                logger.info(
                    "System Status | "
                    f"FPS: {stats['fps']:.1f} | "
                    f"Queue: {stats['queue']}/{self.stream_processor.max_queue_size} | "
                    f"Proc Time: {stats['processing_time']*1000:.1f}ms"
                )
                
        threading.Thread(target=monitor, daemon=True).start()

    def _shutdown(self, *threads):
        """Enhanced graceful shutdown procedure"""
        logger.info("Initiating shutdown...")
        print("\n" + "="*50)
        print("Shutting down Terra Rover...")
        self.running = False
        
        # Stop stream processor first
        self.stream_processor.stop()
        
        # Shutdown parallel processor
        self.parallel_processor.executor.shutdown(wait=False, cancel_futures=True)
        
        # Force terminate any remaining threads
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                logger.warning(f"Terminating lingering thread: {thread.name}")
                try:
                    thread._stop()  # Force stop for stubborn threads
                except:
                    pass
        
        cv2.destroyAllWindows()
        print("Shutdown complete! Thank you for using Terra Rover.")
        print("="*50 + "\n")
        logger.info("System shutdown complete")
        os._exit(0)  # Force exit if normal shutdown fails

if __name__ == "__main__":
    rover = TerraRover()
    rover.start()