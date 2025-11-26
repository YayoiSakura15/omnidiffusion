#!/bin/bash

# OmniFlow 一键环境配置脚本
# 用于在新环境中快速配置 OmniFlow 训练和推理环境

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "                   OmniFlow 环境配置脚本                              "
echo "======================================================================"
echo ""

# ============ 颜色定义 ============
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============ 日志函数 ============
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============ 检查 Python 环境 ============
log_info "检查 Python 环境..."

if ! command -v python3 &> /dev/null; then
    log_error "Python3 未安装，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
log_success "Python 版本: $PYTHON_VERSION"

# ============ 创建/激活虚拟环境 (可选) ============
read -p "是否创建新的虚拟环境? (y/N): " CREATE_VENV
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    VENV_NAME="omniflow_env"
    log_info "创建虚拟环境: $VENV_NAME"
    python3 -m venv $VENV_NAME
    source $VENV_NAME/bin/activate
    log_success "虚拟环境已激活"
else
    log_info "跳过虚拟环境创建"
fi

# ============ 安装依赖 ============
log_info "开始安装 Python 依赖..."

if [ ! -f "requirements.txt" ]; then
    log_error "requirements.txt 未找到！"
    exit 1
fi

log_info "升级 pip..."
pip install --upgrade pip

log_info "安装依赖包 (这可能需要几分钟)..."
pip install -r requirements.txt

log_success "依赖安装完成"

# ============ 安装 OmniFlow 包 ============
log_info "安装 OmniFlow 包..."

if [ -d "jiangqf/OmniFlows" ]; then
    cd jiangqf/OmniFlows
    pip install -e .
    cd ../..
    log_success "OmniFlow 包安装完成"
else
    log_warn "jiangqf/OmniFlows 目录未找到，跳过包安装"
fi

# ============ 创建必要的目录 ============
log_info "创建必要的目录结构..."

mkdir -p jiangqf/OmniFlows/models
mkdir -p jiangqf/OmniFlows/checkpoints
mkdir -p jiangqf/OmniFlows/logs
mkdir -p dataset/data

log_success "目录结构创建完成"

# ============ 下载数据集 ============
echo ""
read -p "是否现在下载数据集? (y/N): " DOWNLOAD_DATA
if [[ "$DOWNLOAD_DATA" =~ ^[Yy]$ ]]; then
    log_info "开始下载数据集..."

    read -p "下载模式: [1] streaming (少量样本，快速) [2] full (完整数据集，慢): " DATA_MODE

    if [ "$DATA_MODE" = "1" ]; then
        log_info "使用 streaming 模式下载前 1000 个样本..."
        python dataset/download_data.py --streaming --max-samples 1000
    elif [ "$DATA_MODE" = "2" ]; then
        log_info "下载完整数据集 (可能需要很长时间)..."
        python dataset/download_data.py
    else
        log_warn "无效选项，跳过数据集下载"
    fi
else
    log_info "跳过数据集下载 (稍后可运行: python dataset/download_data.py)"
fi

# ============ 下载模型 ============
echo ""
read -p "是否现在下载预训练模型? (y/N): " DOWNLOAD_MODELS
if [[ "$DOWNLOAD_MODELS" =~ ^[Yy]$ ]]; then
    log_info "开始下载预训练模型..."

    # 首先安装 huggingface_hub（如果还没有）
    pip install -q huggingface_hub

    python download_models.py --models clip_vit_l t5_large languagebind_audio

    log_info "如需下载完整 OmniFlow 模型，请手动运行:"
    log_info "  python download_models.py --models omniflow"
else
    log_info "跳过模型下载 (稍后可运行: python download_models.py)"
fi

# ============ 环境检查 ============
log_info "检查 CUDA 可用性..."
python3 -c "import torch; print('CUDA 可用:', torch.cuda.is_available()); print('CUDA 设备数:', torch.cuda.device_count()) if torch.cuda.is_available() else None"

# ============ 完成 ============
echo ""
echo "======================================================================"
echo -e "${GREEN}                  环境配置完成！                                ${NC}"
echo "======================================================================"
echo ""
echo "📋 下一步操作:"
echo ""
echo "  1. 下载数据集 (如果还未下载):"
echo "     python dataset/download_data.py --streaming --max-samples 1000"
echo ""
echo "  2. 下载模型 (如果还未下载):"
echo "     python download_models.py"
echo ""
echo "  3. 开始训练:"
echo "     cd jiangqf/OmniFlows"
echo "     ./run_example.sh"
echo ""
echo "  4. 或使用自定义参数训练:"
echo "     python scripts/train_text.py --model_path models/OmniFlow-v0.5 \\"
echo "         --data_config config/data_config.json --batch_size 4"
echo ""
echo "  5. 评估模型:"
echo "     python scripts/eval_text.py --predictions results.json"
echo ""
echo "======================================================================"
echo ""

# 保存环境信息
log_info "保存环境信息到 environment_info.txt"
{
    echo "OmniFlow 环境信息"
    echo "================="
    echo "日期: $(date)"
    echo "Python 版本: $(python3 --version)"
    echo "PyTorch 版本: $(python3 -c 'import torch; print(torch.__version__)')"
    echo "CUDA 可用: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
    echo "GPU 信息:"
    python3 -c "import torch; [print(f'  - {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "  无 GPU"
} > environment_info.txt

log_success "环境信息已保存"
echo ""
