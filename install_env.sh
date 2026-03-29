#!/bin/bash
set -e  # 遇到错误立即退出

echo ">>> 开始配置环境..."

# 1. 设置 Pip 镜像源 (清华源)
echo ">>> 配置 Pip 使用清华源..."
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 安装基础依赖
# ninja 通常用于加速 CUDA 编译
echo ">>> 安装基础依赖 (opencv-python, matplotlib, ninja)..."
pip install opencv-python matplotlib ninja

# 3. 安装 GroundingDINO
echo ">>> 安装 GroundingDINO (使用 gitclone.com 加速)..."
if [ ! -d "GroundingDINO" ]; then
    git clone https://gitclone.com/github.com/IDEA-Research/GroundingDINO.git
    echo "GroundingDINO 代码库克隆完成。"
else
    echo "GroundingDINO 目录已存在，跳过克隆。"
fi

cd GroundingDINO
# GroundingDINO 需要编译 CUDA 算子，可能需要一定时间
echo "正在安装 GroundingDINO (pip install -e .)..."
pip install -e .
cd ..

# 4. 安装 Segment Anything (SAM)
echo ">>> 安装 Segment Anything (SAM) (使用 gitclone.com 加速)..."
if [ ! -d "segment-anything" ]; then
    git clone https://gitclone.com/github.com/facebookresearch/segment-anything.git
    echo "segment-anything 代码库克隆完成。"
else
    echo "segment-anything 目录已存在，跳过克隆。"
fi

cd segment-anything
echo "正在安装 SAM (pip install -e .)..."
pip install -e .
cd ..

echo ">>> 环境配置完成！"
