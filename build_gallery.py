import os
import cv2
import psycopg2
import numpy as np
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# 假设已经有了 core_pipeline.py 中的 CatIdentificationPipeline 类
try:
    from core_pipeline import CatIdentificationPipeline
except ImportError:
    print("Warning: core_pipeline.py not found. Assuming it will be available in the execution environment.")
    # Mocking for local development if needed, but in production this should fail or rely on the file being present
    class CatIdentificationPipeline:
        def __init__(self):
            pass
        def process_image(self, path):
            pass

def build_gallery():
    # 1. 连接数据库
    try:
        conn = psycopg2.connect(
            dbname="cat_recognition",
            user="postgres",
            password="123456",
            host="localhost"
        )
        # conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return

    # 2. 初始化 Pipeline
    print("Initializing pipeline...")
    try:
        pipeline = CatIdentificationPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    root_dir = "dataset/gallery"
    output_dir = "output/crops"
    
    if not os.path.exists(root_dir):
        print(f"Error: Dataset directory {root_dir} not found.")
        return

    # 获取所有图片总数用于进度显示
    total_images = 0
    for root, dirs, files in os.walk(root_dir):
        total_images += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {total_images} images in {root_dir}")
    
    count = 0
    
    # 3. 遍历目录
    for root, dirs, files in os.walk(root_dir):
        # 提取猫的名字 (从文件夹名字)
        cat_name = os.path.basename(root)
        
        # 跳过根目录如果它不包含猫的名字 (即 dataset/gallery 本身)
        if root == root_dir:
            continue
            
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            image_path = os.path.join(root, file)
            count += 1
            
            print(f"Processing {cat_name} ({count}/{total_images}): {file}...")
            
            try:
                # 4. 调用 pipeline 处理图片
                # 假设返回 vector (numpy array) 和 cat_crop (image array)
                result = pipeline.process_image(image_path)
                
                if result is None:
                    print(f"  - Failed to detect/process: {file}")
                    continue
                    
                vector, cat_crop = result
                
                # 5. 存入数据库
                # 将 numpy array 转换为 list 以便 pgvector 存储
                embedding_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
                
                insert_query = """
                INSERT INTO cat_gallery (cat_name, image_path, embedding)
                VALUES (%s, %s, %s)
                """
                cursor.execute(insert_query, (cat_name, image_path, embedding_list))
                
                # 6. 保存 Crop 图
                save_dir = os.path.join(output_dir, cat_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, cat_crop)
                
                # print(f"  - Saved crop to {save_path}")
                
            except Exception as e:
                print(f"  - Error processing {file}: {e}")
                conn.rollback() # 出错回滚
                continue
                
            # 每处理一张提交一次，或者批量提交，这里选择简单起见每张提交
            conn.commit()

    print("Gallery build complete!")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    build_gallery()
