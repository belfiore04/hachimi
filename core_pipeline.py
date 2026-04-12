"""
core_pipeline.py
Complete cat identification pipeline: Detect → Segment → DINOv2 Feature Extraction
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from cat_pipeline import CatPipeline


class CatIdentificationPipeline:
    def __init__(self, gd_config_path, gd_checkpoint_path, sam_checkpoint_path,
                 sam_type="vit_h", device=None):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"Using device: {self.device}")

        # Stage 1 & 2: Detection + Segmentation
        self.cat_pipeline = CatPipeline(
            gd_config_path=gd_config_path,
            gd_checkpoint_path=gd_checkpoint_path,
            sam_checkpoint_path=sam_checkpoint_path,
            sam_type=sam_type,
            device=self.device,
        )

        # Stage 3: DINOv2 feature extraction
        print("Loading DINOv2 (vitb14)...")
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.dinov2.to(self.device)
        self.dinov2.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        print("All models loaded.")

    def extract_feature(self, masked_image_np):
        """Extract 768-dim DINOv2 feature from a masked image (numpy RGB)."""
        pil_image = Image.fromarray(masked_image_np)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.dinov2(tensor)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.squeeze().cpu().tolist()

    def process_image(self, image_path):
        """
        Full pipeline: detect → segment → extract feature.
        Returns (feature_vector, masked_image) or None if no cat detected.
        """
        image_source, box = self.cat_pipeline.detect(image_path)
        if box is None:
            return None

        masked_image, mask = self.cat_pipeline.segment(image_source, box)
        if masked_image is None:
            return None

        feature_vector = self.extract_feature(masked_image)
        return feature_vector, masked_image
