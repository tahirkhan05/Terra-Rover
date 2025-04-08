import torch
from ultralytics import YOLO
from utils.logger import logger

def safe_load_model(model_path):
    """Safely load YOLO model with PyTorch 2.6+ compatibility"""
    try:
        # First try normal load
        return YOLO(model_path)
    except Exception as e:
        logger.warning(f"Standard load failed, attempting workaround: {str(e)}")
        # Workaround for PyTorch 2.6+ weights_only issue
        import torch.serialization
        from ultralytics.nn.tasks import DetectionModel
        
        # Add DetectionModel to safe globals
        torch.serialization.add_safe_globals([DetectionModel])
        
        # Try loading again
        return YOLO(model_path)