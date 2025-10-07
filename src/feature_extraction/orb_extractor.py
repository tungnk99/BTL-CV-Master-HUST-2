import cv2
import numpy as np
from typing import Optional
from .base_extractor import BaseFeatureExtractor


class ORBExtractor(BaseFeatureExtractor):
    def __init__(
            self,
            nfeatures: int = 500,
            scaleFactor: float = 1.05,
            nlevels: int = 4,
            edgeThreshold: int = 5,
            firstLevel: int = 0,
            WTA_K: int = 2,
            scoreType: int = cv2.ORB_HARRIS_SCORE,
            patchSize: int = 15,
            fastThreshold: int = 10,
            **kwargs):
        """
            nfeatures: Số lượng feature tối đa muốn detect
            scaleFactor: Hệ số scale giữa các pyramid levels. Nếu ảnh nhỏ thì nên giảm xuống (1.05 - 1.1) để có nhiều level chi tiết hơn.
            nlevels: Số tầng pyramid scale. Ảnh 32×32 nên để 3-4 tầng
            edgeThreshold: Kích thước vùng bỏ qua biên khi detect.  Ảnh 32×32 nên để 3-5
            patchSize: Kích thước patch mô tả. Ảnh 32×32 nên giảm còn 15 hoặc 9
            fastThreshold: Ngưỡng FAST (độ nhạy corner). Giảm xuống (5–10) để phát hiện nhiều điểm hơn trong ảnh mờ
        """
        super().__init__(**kwargs)
        
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        self.edgeThreshold = edgeThreshold
        self.firstLevel = firstLevel
        self.WTA_K = WTA_K
        self.scoreType = scoreType
        self.patchSize = patchSize
        self.fastThreshold = fastThreshold
        
        # Initialize ORB detector
        self.detector = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            firstLevel=firstLevel,
            WTA_K=WTA_K,
            scoreType=scoreType,
            patchSize=patchSize,
            fastThreshold=fastThreshold
        )
    
    def extract_features(self, image: np.ndarray, image_name: str = "") -> Optional[np.ndarray]:
        # Check if image is valid
        if image is None or image.size == 0:
            print(f"Warning: Empty or invalid image: {image_name}")
            return np.array([]).reshape(0, 32)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume image is in range [0, 1] and convert to [0, 255]
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to grayscale if necessary
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            gray = image[:, :, 0]
        elif len(image.shape) == 2:
            gray = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Ensure grayscale image is uint8
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        if self.debug:
            image_name = image_name or "img.png"
            feature_img = cv2.drawKeypoints(gray, keypoints, image)
            cv2.imwrite(f"{self.log_dir}/orb_{image_name}", feature_img)


        if descriptors is None:
            return np.array([]).reshape(0, 32)
        
        return descriptors
    
    def get_feature_dim(self) -> int:
        return 32
