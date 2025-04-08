import cv2
import numpy as np
import torch
from model_loader import safe_load_model
from utils.logger import logger

class ObjectDetector:
    def __init__(self, model_path):
        self.model = safe_load_model(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        logger.info(f"Object detection model loaded on {self.device}")
    
    def detect_objects(self, frame):
        try:
            # Use half precision for faster inference if GPU available
            with torch.amp.autocast(device_type=self.device, enabled=self.device == 'cuda'):
                results = self.model(frame, verbose=False)
            
            detected_objects = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]
                    
                    detected_objects.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            return detected_objects, results[0].plot()
        except Exception as e:
            logger.error(f"Object detection error: {str(e)}")
            return [], frame
