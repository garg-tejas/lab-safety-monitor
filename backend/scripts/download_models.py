"""Download required YOLO models."""
import sys
import os
from pathlib import Path
from loguru import logger

# Set environment variable for PyTorch to use legacy loading
os.environ['TORCH_LOAD_WEIGHTS_ONLY'] = 'False'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import settings


def download_models():
    """Download YOLOv8 models."""
    try:
        from ultralytics import YOLO
        
        logger.info("Downloading YOLOv8 models...")
        
        # Download person detection model (YOLOv8 nano)
        logger.info("Downloading YOLOv8n (person detection)...")
        model = YOLO('yolov8n.pt')
        logger.info("YOLOv8n downloaded successfully")
        
        # For PPE model, you would need to train or download a custom model
        logger.info("""
        ===================================================
        IMPORTANT: PPE Detection Model
        ===================================================
        
        The PPE detection model needs to be trained on a custom dataset.
        
        Options:
        1. Train your own model using datasets from:
           - Roboflow: https://universe.roboflow.com/search?q=ppe
           - Kaggle: Hard Hat Workers Dataset
           
        2. Use a pre-trained model if available
        
        To train:
        1. Prepare dataset in YOLO format
        2. Create a YAML config file (see configs/ppe_dataset.yaml.example)
        3. Run: yolo train data=configs/ppe_dataset.yaml model=yolov8n.pt epochs=100
        
        For demo purposes, the system will use mock PPE detection.
        ===================================================
        """)
        
        logger.info("Model download complete!")
        
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        raise


if __name__ == "__main__":
    download_models()
