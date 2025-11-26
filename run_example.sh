#!/bin/bash

# OmniFlow 文本 Decoder Head 训练示例脚本
# 在仓库根目录 OmniFlows 下运行：
#   chmod +x run_example.sh
#   ./run_example.sh

# ========= 配置区域 =========
MODEL_PATH="models/OmniFlow-v0.5"
DATA_CONFIG="config/data_config.json"
OUTPUT_DIR="checkpoints/text_decoder_head_test"
BATCH_SIZE=4
NUM_EPOCHS=1
LR=1e-4
HEAD_DIM=64
# ==========================

echo "=================================="
echo "OmniFlow Text Decoder Head Training"
echo "=================================="
echo ""

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
    echo "Please update MODEL_PATH in this script"
    exit 1
fi

# 检查数据配置
if [ ! -f "$DATA_CONFIG" ]; then
    echo "Error: Data config not found: $DATA_CONFIG"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

###############################################
# 方案 1：只用 MLP Text Decoder Head（无 VQ） #
###############################################
echo ""
echo "[Run] Training with MLP text decoder head (no VQ codebook)..."
echo ""

python scripts/train_text.py \
    --model_path "$MODEL_PATH" \
    --data_config "$DATA_CONFIG" \
    --use_text_decoder_head \
    --text_decoder_head_dim $HEAD_DIM \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --output_dir "$OUTPUT_DIR/mlp_head" \
    --log_interval 50 \
    --eval_interval 500 \
    --save_interval 1000

###################################################
# 方案 2：使用带 VQ codebook 的 Text Decoder Head  #
# 如果暂时不想跑，可以把下面这一段整体注释掉        #
###################################################
# echo ""
# echo "[Run] Training with VQ text decoder head..."
# echo ""
#
# python scripts/train_text.py \
#     --model_path "$MODEL_PATH" \
#     --data_config "$DATA_CONFIG" \
#     --use_text_decoder_head \
#     --use_vq_codebook \
#     --text_decoder_head_dim $HEAD_DIM \
#     --batch_size $BATCH_SIZE \
#     --num_epochs $NUM_EPOCHS \
#     --lr $LR \
#     --output_dir "$OUTPUT_DIR/vq_head" \
#     --log_interval 50 \
#     --eval_interval 500 \
#     --save_interval 1000
#
# echo ""
# echo "=================================="
# echo "All Decoder Head Training Finished"
# echo "Checkpoints:"
# echo "  MLP head -> $OUTPUT_DIR/mlp_head"
# echo "  VQ  head -> $OUTPUT_DIR/vq_head"
# echo "=================================="
# echo ""
# echo "示例测试命令（MLP head）："
# echo "python scripts/test_text.py --model_path $MODEL_PATH --checkpoint $OUTPUT_DIR/mlp_head/best_*.pt ..."
# echo ""
# echo "示例测试命令（VQ head）："
# echo "python scripts/test_text.py --model_path $MODEL_PATH --checkpoint $OUTPUT_DIR/vq_head/best_*.pt ..."
