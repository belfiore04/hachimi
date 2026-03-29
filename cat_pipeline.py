import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# GroundingDINO imports
# Assuming GroundingDINO is installed in the python environment
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything imports
from segment_anything import sam_model_registry, SamPredictor

class CatPipeline:
    def __init__(
        self, 
        gd_config_path, 
        gd_checkpoint_path, 
        sam_checkpoint_path, 
        sam_type="vit_h", 
        device="cuda"
    ):
        self.device = device
        
        print(f"Loading GroundingDINO from {gd_checkpoint_path}...")
        self.gd_model = load_model(gd_config_path, gd_checkpoint_path, device=self.device)
        
        print(f"Loading SAM ({sam_type}) from {sam_checkpoint_path}...")
        self.sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)
        
        print("Models loaded successfully.")

    def detect(self, image_path, text_prompt="cat", box_threshold=0.35, text_threshold=0.25):
        """
        Use GroundingDINO to detect 'cat' (or other text_prompt).
        Returns:
            image_source: Original image (H, W, 3) numpy array (BGR for cv2 or RGB)
            best_box: The bounding box with highest confidence [cx, cy, w, h] normalized
        """
        # load_image returns (image_source, image)
        # image_source is numpy array (H, W, 3) RGB via PIL
        # image is Tensor (1, 3, H, W) normalized
        image_source, image = load_image(image_path)
        
        boxes, logits, phrases = predict(
            model=self.gd_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        if len(boxes) == 0:
            print(f"No objects found for prompt: {text_prompt}")
            return image_source, None

        # Find the box with the highest confidence prediction
        # logits is a tensor of scores
        best_idx = torch.argmax(logits)
        best_box = boxes[best_idx]  # [cx, cy, w, h] normalized
        
        # print(f"Detected {len(boxes)} boxes. Best score: {logits[best_idx]:.4f}")
        
        return image_source, best_box

    def segment(self, image_source, box):
        """
        Use SAM to segment the object defined by box.
        Args:
            image_source: numpy array (H, W, 3), RGB
            box: normalized box [cx, cy, w, h]
        Returns:
            masked_image: numpy array (H, W, 3) with background blackened
            mask: boolean mask
        """
        if box is None:
            return None, None
            
        # GroundingDINO box is [cx, cy, w, h] normalized
        # SAM expects [x1, y1, x2, y2] absolute pixels
        h, w, _ = image_source.shape
        
        cx, cy, bw, bh = box.unbind(0)
        cx, cy, bw, bh = cx.item(), cy.item(), bw.item(), bh.item()
        
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        
        input_box = np.array([x1, y1, x2, y2])

        self.sam_predictor.set_image(image_source)
        
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        # masks shape: (1, H, W)
        mask = masks[0]
        
        # Apply mask to image (black background for non-mask areas)
        # image_source is RGB. 
        masked_image = image_source.copy()
        masked_image[~mask] = [0, 0, 0]
        
        return masked_image, mask

    def save_crop(self, image, save_path):
        """
        Save the image to disk.
        Args:
            image: numpy array (RGB)
            save_path: path to save
        """
        if image is None:
            print("No image to save.")
            return

        # Convert RGB to BGR for cv2 saving, or use PIL
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image_bgr)
        print(f"Saved crop to {save_path}")

# Example usage (commented out)
# if __name__ == "__main__":
#     pipeline = CatPipeline(...)
#     img, box = pipeline.detect("cat.jpg")
#     res, mask = pipeline.segment(img, box)
#     pipeline.save_crop(res, "cat_crop.jpg")
