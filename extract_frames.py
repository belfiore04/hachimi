import cv2
import os
import glob
import re

def get_next_sequence_id(target_base_dir):
    """
    检查目标目录下已有的序列号文件夹 (cat_001, cat_002...)，返回下一个可用的序号。
    例如：如果没有文件夹 -> 返回 1
          如果有 cat_001, cat_002 -> 返回 3
    """
    if not os.path.exists(target_base_dir):
        return 1
    
    max_id = 0
    # 遍历直接子目录
    for name in os.listdir(target_base_dir):
        # 检查是否匹配 cat_xxx 格式
        match = re.match(r'cat_(\d+)', name)
        if os.path.isdir(os.path.join(target_base_dir, name)) and match:
            try:
                current_id = int(match.group(1))
                if current_id > max_id:
                    max_id = current_id
            except ValueError:
                continue
    
    return max_id + 1

def extract_frames():
    # 配置
    SOURCE_DIR = 'dataset/gallery/videos'  # 视频所在文件夹
    TARGET_BASE_DIR = 'dataset/gallery' # 修正 spelling: gallery, 且直接包含 cat_xxx 文件夹
    FRAME_INTERVAL = 40
    
    # 支持的视频扩展名
    VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']

    # 确保源目录存在
    if not os.path.exists(SOURCE_DIR):
        print(f"警告: 源目录 '{SOURCE_DIR}' 不存在。正在尝试创建一个空目录...")
        try:
            os.makedirs(SOURCE_DIR)
            print(f"已创建 '{SOURCE_DIR}'，请将视频文件放入其中后重新运行脚本。")
        except OSError as e:
            print(f"创建目录失败: {e}")
        return

    # 获取所有视频文件
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
    
    if not video_files:
        print(f"在 '{SOURCE_DIR}' 中未找到视频文件。请放入视频文件后重试。")
        return

    print(f"找到 {len(video_files)} 个视频文件，准备开始处理...")

    # 获取起始序号
    current_seq_id = get_next_sequence_id(TARGET_BASE_DIR)
    
    for video_path in video_files:
        print(f"\n正在处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            continue
            
        # 创建对应的输出文件夹 (例如 cat_001, cat_002)
        folder_name = f"cat_{current_seq_id:03d}"
        output_dir = os.path.join(TARGET_BASE_DIR, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"  -> 输出目录: {output_dir}")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每隔 FRAME_INTERVAL 帧保存一次
            if frame_count % FRAME_INTERVAL == 0:
                # 图片命名: frame_00000.jpg
                image_name = f"frame_{saved_count:05d}.jpg"
                image_path = os.path.join(output_dir, image_name)
                
                try:
                    cv2.imwrite(image_path, frame)
                    saved_count += 1
                except Exception as e:
                    print(f"    保存图片失败: {e}")

            frame_count += 1
        
        cap.release()
        print(f"  -> 完成。共保存 {saved_count} 张图片。")
        
        # 处理完一个视频，序号 +1
        current_seq_id += 1

    print("\n所有视频处理完毕！")

if __name__ == "__main__":
    extract_frames()
