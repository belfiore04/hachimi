import cv2
import numpy as np
import os

def create_dummy_video(filename, width=640, height=480, fps=10, duration_sec=5):
    os.makedirs('videos', exist_ok=True)
    filepath = os.path.join('videos', filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    frames = fps * duration_sec
    for i in range(frames):
        # Create a frame with a changing color
        img = np.zeros((height, width, 3), dtype=np.uint8)
        color_val = int((i / frames) * 255)
        img[:] = (color_val, 255 - color_val, 128)
        
        # Add text
        cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(img)
    
    out.release()
    print(f"Created {filepath}")

if __name__ == "__main__":
    create_dummy_video('test_video_1.mp4')
    create_dummy_video('test_video_2.mp4')
