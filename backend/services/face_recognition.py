"""Face detection and recognition service."""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger
import sys

sys.path.append('..')
from config import settings


class FaceRecognitionService:
    """Service for face detection and recognition."""
    
    def __init__(self):
        """Initialize face detection models."""
        self.embedding_dim = settings.FACE_EMBEDDING_DIM
        self.recognition_threshold = settings.FACE_RECOGNITION_THRESHOLD
        
        # Try to load FaceNet model
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
            
            # MTCNN for face detection
            self.mtcnn = MTCNN(
                keep_all=True,
                device=self.device,
                post_process=False
            )
            
            # InceptionResnetV1 for face embeddings
            self.resnet = InceptionResnetV1(
                pretrained=settings.FACENET_MODEL
            ).eval().to(self.device)
            
            logger.info(f"Face recognition models loaded on {self.device}")
            self.models_available = True
            
        except Exception as e:
            logger.warning(f"Could not load FaceNet models: {e}. Using fallback detection.")
            self.models_available = False
            self._init_fallback_detector()
    
    def _init_fallback_detector(self):
        """Initialize OpenCV fallback face detector."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("Using OpenCV Haar Cascade for face detection")
        except Exception as e:
            logger.error(f"Could not initialize face detector: {e}")
            self.face_cascade = None
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame.
        
        Args:
            frame: Input image frame (BGR format from OpenCV)
            
        Returns:
            List of face detections with bounding boxes and embeddings
        """
        if self.models_available:
            return self._detect_faces_mtcnn(frame)
        else:
            return self._detect_faces_fallback(frame)
    
    def _detect_faces_mtcnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using MTCNN."""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            boxes, probs, landmarks = self.mtcnn.detect(rgb_frame, landmarks=True)
            
            if boxes is None:
                return []
            
            faces = []
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.9:  # Confidence threshold
                    continue
                
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Extract face region for embedding
                face_img = rgb_frame[y1:y2, x1:x2]
                embedding = self.get_face_embedding(face_img)
                
                faces.append({
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    },
                    'confidence': float(prob),
                    'embedding': embedding.tolist() if embedding is not None else None,
                    'landmarks': landmarks[i].tolist() if landmarks is not None else None
                })
            
            return faces
        
        except Exception as e:
            logger.error(f"Error in MTCNN face detection: {e}")
            return []
    
    def _detect_faces_fallback(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascade."""
        if self.face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': {
                        'x1': float(x),
                        'y1': float(y),
                        'x2': float(x + w),
                        'y2': float(y + h)
                    },
                    'confidence': 0.9,  # Default confidence for Haar Cascade
                    'embedding': self._generate_mock_embedding(),
                    'landmarks': None
                })
            
            return detections
        
        except Exception as e:
            logger.error(f"Error in fallback face detection: {e}")
            return []
    
    def get_face_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding from face image.
        
        Args:
            face_img: Face image (RGB format)
            
        Returns:
            512-dimensional embedding vector
        """
        if not self.models_available:
            return self._generate_mock_embedding()
        
        try:
            import torch
            from torchvision import transforms
            
            # Resize and normalize face
            face_img = cv2.resize(face_img, (160, 160))
            face_tensor = transforms.ToTensor()(face_img).unsqueeze(0)
            face_tensor = (face_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.resnet(face_tensor.to(self.device))
            
            return embedding.cpu().numpy().flatten()
        
        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            return None
    
    def _generate_mock_embedding(self) -> np.ndarray:
        """Generate mock embedding for testing."""
        return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def compare_faces(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Compare two face embeddings using cosine similarity.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Tuple of (similarity_score, is_match)
        """
        try:
            # Convert to numpy arrays if needed
            if isinstance(embedding1, list):
                embedding1 = np.array(embedding1)
            if isinstance(embedding2, list):
                embedding2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            is_match = similarity >= self.recognition_threshold
            
            return float(similarity), is_match
        
        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return 0.0, False
    
    def find_matching_person(
        self,
        face_embedding: np.ndarray,
        known_persons: List[Dict]
    ) -> Optional[str]:
        """
        Find matching person from known persons database.
        
        Args:
            face_embedding: Embedding of detected face
            known_persons: List of dicts with 'person_id' and 'face_embedding'
            
        Returns:
            person_id if match found, None otherwise
        """
        best_match_id = None
        best_similarity = 0.0
        
        for person in known_persons:
            if person.get('face_embedding') is None:
                continue
            
            similarity, is_match = self.compare_faces(
                face_embedding,
                person['face_embedding']
            )
            
            if is_match and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person['person_id']
        
        return best_match_id
