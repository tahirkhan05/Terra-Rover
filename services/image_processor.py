import cv2
import os
import time
from config.settings import settings
from utils.logger import logger

class ImageProcessor:
    def __init__(self):
        os.makedirs(settings.LOCAL_SAVE_PATH, exist_ok=True)
    
    def save_frame_locally(self, frame):
        try:
            timestamp = int(time.time() * 1000)  # Milliseconds for unique filename
            filename = f"{settings.LOCAL_SAVE_PATH}/frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            return filename
        except Exception as e:
            logger.error(f"Frame save error: {str(e)}")
            return None
    
    def display_frame(self, frame, window_name='Terra Rover'):
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
    
    def process_and_store_frame(self, frame):
        """Process and store captured frame with verification"""
        from services.aws_client import AWSClient
        import cv2
        import time
        
        # Save locally
        local_path = self.save_frame_locally(frame)
        if not local_path:
            return None, None
        
        # Compress for S3 upload
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            logger.error("Failed to compress frame")
            return None, None
            
        # Upload to S3
        aws_client = AWSClient()
        timestamp = int(time.time() * 1000)
        key = f"frames/frame_{timestamp}.jpg"
        
        try:
            aws_client.s3.put_object(
                Bucket=settings.S3_BUCKET,
                Key=key,
                Body=buffer.tobytes(),
                ContentType='image/jpeg'
            )
            
            # Verify upload
            aws_client.s3.head_object(Bucket=settings.S3_BUCKET, Key=key)
            logger.debug(f"Successfully uploaded frame to s3://{settings.S3_BUCKET}/{key}")
            return f"s3://{settings.S3_BUCKET}/{key}", key
        except Exception as e:
            logger.error(f"Failed to upload frame to S3: {str(e)}")
            return None, None