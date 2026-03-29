import os
import argparse
from cat_pipeline import CatPipeline

def main():
    parser = argparse.ArgumentParser(description="Test CatPipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="output_crop.jpg", help="Path to save output")
    parser.add_argument("--gd_config", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Path to GroundingDINO config")
    parser.add_argument("--gd_checkpoint", type=str, default="weights/groundingdino_swint_ogc.pth", help="Path to GroundingDINO checkpoint")
    parser.add_argument("--sam_checkpoint", type=str, default="weights/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.gd_checkpoint):
        print(f"Error: GD Checkpoint not found at {args.gd_checkpoint}")
        return
    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: SAM Checkpoint not found at {args.sam_checkpoint}")
        return
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    pipeline = CatPipeline(
        gd_config_path=args.gd_config,
        gd_checkpoint_path=args.gd_checkpoint,
        sam_checkpoint_path=args.sam_checkpoint
    )

    print(f"Procesing {args.image}...")
    image_source, box = pipeline.detect(args.image)
    
    if box is None:
        print("No cat detected.")
        return

    masked_image, mask = pipeline.segment(image_source, box)
    pipeline.save_crop(masked_image, args.output)
    print("Test Complete!")

if __name__ == "__main__":
    main()
