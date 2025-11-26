# OmniFlow 训练和推理环境

完整的 OmniFlow 多模态生成模型训练和推理环境配置指南。

---

## 📋 目录

- [目录结构](#-目录结构)
- [快速开始](#-快速开始-5分钟)
- [环境配置](#-环境配置)
- [数据集下载](#-数据集下载)
- [模型下载](#-模型下载)
- [训练模型](#-训练模型)
- [实验配置](#-实验配置)
- [评估模型](#-评估模型)
- [常见问题](#-常见问题)

---

## 📁 目录结构

```
qf/
├── dataset/                      # 数据集目录
│   ├── download_data.py         # 数据下载脚本
│   └── data/                    # 下载后的数据
└── jiangqf/OmniFlows/           # OmniFlow 主项目（当前目录）
    ├── README.md                # 本文件
    ├── setup.sh                 # 一键环境配置
    ├── download_models.py       # 模型下载
    ├── requirements.txt         # Python 依赖
    ├── run_example.sh           # 快速训练示例
    ├── models/                  # 预训练模型
    ├── checkpoints/             # 训练检查点
    ├── logs/                    # 训练日志
    ├── omniflow/                # 核心代码
    ├── scripts/                 # 训练和评估脚本
    │   ├── train_text.py       # 文本解码器训练
    │   ├── eval_text.py        # 文本评估
    │   └── generate_text.py    # 文本生成
    └── config/                  # 配置文件
        └── data_config.json    # 数据配置
```

---

## 🚀 快速开始 (5分钟)

### 步骤 1: 配置环境（2 分钟）

```bash
# 进入项目目录
cd jiangqf/OmniFlows

# 运行一键配置脚本
./setup.sh

# 或手动安装
pip install -r requirements.txt && pip install -e .
```

### 步骤 2: 准备数据（1 分钟）

```bash
# 回到根目录下载测试数据（streaming 模式，仅 1000 个样本）
cd ../../
python dataset/download_data.py --streaming --max-samples 1000
cd jiangqf/OmniFlows
```

### 步骤 3: 开始训练（2 分钟）

```bash
# 运行示例训练脚本
./run_example.sh
```

完成！🎉

---

## 🔧 环境配置

### 自动配置（推荐）

```bash
./setup.sh
```

脚本会自动完成：
- ✅ 检查 Python 环境
- ✅ 创建虚拟环境（可选）
- ✅ 安装所有依赖
- ✅ 创建必要目录
- ✅ 可选：下载数据和模型

### 手动配置

#### 1. 创建虚拟环境（推荐）

```bash
python3 -m venv omniflow_env
source omniflow_env/bin/activate  # Linux/Mac
```

#### 2. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. 安装 OmniFlow 包

```bash
pip install -e .
```

---

## 📦 数据集下载

### 快速测试（Streaming 模式，推荐）

下载少量样本用于快速测试：

```bash
cd ../../  # 回到 qf/ 根目录
python dataset/download_data.py --streaming --max-samples 1000
cd jiangqf/OmniFlows
```

下载内容：
- WikiText-103: 前 1000 个样本
- COCO Caption: 前 1000 个样本
- AudioCaps: 前 1000 个样本
- LLaVA-CC3M: 前 1000 个样本

### 下载完整数据集

```bash
cd ../../
python dataset/download_data.py  # 下载所有数据集
cd jiangqf/OmniFlows
```

### 下载特定数据集

```bash
cd ../../

# 只下载 WikiText
python dataset/download_data.py --wikitext

# 下载多个数据集
python dataset/download_data.py --coco --audiocaps

# 下载 10% 子集
python dataset/download_data.py --subset-fraction 0.1

cd jiangqf/OmniFlows
```

### 数据集说明

| 数据集 | HuggingFace ID | 用途 | 模态 |
|--------|---------------|------|------|
| WikiText-103 | `Salesforce/wikitext` | 纯文本训练 | Text |
| COCO Caption | `lmms-lab/COCO-Caption` | 图像描述 | Image + Text |
| AudioCaps | `OpenSound/AudioCaps` | 音频描述 | Audio + Text |
| LLaVA-CC3M | `liuhaotian/LLaVA-CC3M-Pretrain-595K` | 视觉对话 | Image + Text |

数据保存位置：`../../dataset/data/`

---

## 🤖 模型下载

### 使用下载脚本

```bash
# 下载所有必需模型
python download_models.py

# 只下载特定模型
python download_models.py --models clip_vit_l t5_large

# 强制重新下载
python download_models.py --force
```

### 需要的模型

模型自动下载到 `models/` 目录：

1. **CLIP-ViT-L-14** (~1.7 GB)
   - HF ID: `laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K`
   - 用途：图像和文本编码器

2. **T5-Large** (~2.8 GB)
   - HF ID: `google/flan-t5-large`
   - 用途：文本编码器

3. **LanguageBind Audio** (~1.4 GB)
   - HF ID: `LanguageBind/LanguageBind_Audio_FT`
   - 用途：音频编码器

4. **OmniFlow 主模型** (需要手动配置)
   - 包含：Transformer + VAE + Text VAE
   - 位置：`models/OmniFlow-v0.5/`

---

## 🎯 训练模型

### 使用示例脚本（最简单）

```bash
./run_example.sh
```

默认配置：
- MLP Text Decoder Head
- 1 epoch
- Batch size: 4
- Learning rate: 1e-4
- 输出：`checkpoints/text_decoder_head_test/mlp_head/`

### 自定义训练参数

```bash
python scripts/train_text.py \
    --model_path models/OmniFlow-v0.5 \
    --data_config config/data_config.json \
    --use_text_decoder_head \
    --text_decoder_head_dim 2048 \
    --batch_size 8 \
    --num_epochs 5 \
    --lr 2e-4 \
    --output_dir checkpoints/my_training \
    --log_interval 100 \
    --eval_interval 500 \
    --save_interval 2000
```

### 重要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 预训练模型路径 | 必需 |
| `--data_config` | 数据配置文件 | 必需 |
| `--use_text_decoder_head` | 使用文本解码头 | False |
| `--use_vq_codebook` | 使用 VQ codebook | False |
| `--text_decoder_head_dim` | 解码头输出维度 | 2048 |
| `--use_latent_refiner` | 使用 LatentRefiner | False |
| `--batch_size` | 批大小 | 16 |
| `--num_epochs` | 训练轮数 | 3 |
| `--lr` | 学习率 | 1e-4 |

---

## 🧪 实验配置

### Baseline：简单 MLP Head

最基础配置，无 VQ，无 Refiner。

```bash
python scripts/train_text.py \
  --model_path ./models/OmniFlow-v0.5 \
  --data_config ./config/data_config.json \
  --use_text_decoder_head \
  --text_decoder_head_dim 64 \
  --batch_size 8 \
  --num_epochs 1 \
  --output_dir ./checkpoints/baseline_head
```

**特点**:
- 输出维度：64-d latent
- 参数量：最少
- 训练速度：最快

### 实验 A：MLP Head + LatentRefiner

在 baseline 基础上添加 LatentRefiner MLP 优化表征。

```bash
python scripts/train_text.py \
  --model_path ./models/OmniFlow-v0.5 \
  --data_config ./config/data_config.json \
  --use_text_decoder_head \
  --text_decoder_head_dim 64 \
  --use_latent_refiner \
  --latent_refiner_hidden_dim 256 \
  --latent_refiner_layers 2 \
  --batch_size 8 \
  --num_epochs 1 \
  --output_dir ./checkpoints/head_plus_refiner
```

**特点**:
- 输出维度：64-d latent
- 额外参数：~50K (LatentRefiner)
- LatentRefiner 结构：Input → MLP(256) → MLP(256) → Output + Residual

### 实验 B：VQ Codebook Head

使用 VQ (Vector Quantization) codebook 减少连续-离散映射误差。

```bash
python scripts/train_text.py \
  --model_path ./models/OmniFlow-v0.5 \
  --data_config ./config/data_config.json \
  --use_text_decoder_head \
  --use_vq_codebook \
  --text_decoder_head_dim 64 \
  --batch_size 8 \
  --num_epochs 1 \
  --output_dir ./checkpoints/vq_head
```

**特点**:
- 输出维度：64-d latent
- VQ Codebook 大小：8192
- 额外损失：VQ loss

### 实验 A+B：VQ Head + LatentRefiner

完整配置，结合 VQ 和 Refiner。

```bash
python scripts/train_text.py \
  --model_path ./models/OmniFlow-v0.5 \
  --data_config ./config/data_config.json \
  --use_text_decoder_head \
  --use_vq_codebook \
  --text_decoder_head_dim 64 \
  --use_latent_refiner \
  --latent_refiner_hidden_dim 256 \
  --latent_refiner_layers 2 \
  --batch_size 8 \
  --num_epochs 1 \
  --output_dir ./checkpoints/vq_head_plus_refiner
```

**特点**:
- VQ quantization + Latent refinement
- 参数量：最多
- 理论上表现最好

---

## 📊 评估模型

### 1. 生成预测结果

```bash
python scripts/generate_text.py \
    --model_path models/OmniFlow-v0.5 \
    --checkpoint checkpoints/baseline_head/best_*.pt \
    --data_config config/data_config.json \
    --output predictions.json \
    --batch_size 4
```

### 2. 计算评估指标

```bash
python scripts/eval_text.py \
    --predictions predictions.json \
    --metrics bleu rouge meteor \
    --output_json eval_results.json
```

支持指标：
- **BLEU-4**: 文本生成质量
- **ROUGE-L**: 文本相似度
- **METEOR**: 综合评估

### 3. 对比实验

评估所有实验配置并对比：

```bash
# Baseline
python scripts/generate_text.py \
  --checkpoint ./checkpoints/baseline_head/best_*.pt \
  --output ./results/baseline_predictions.json

# MLP + Refiner
python scripts/generate_text.py \
  --checkpoint ./checkpoints/head_plus_refiner/best_*.pt \
  --output ./results/refiner_predictions.json

# VQ Head
python scripts/generate_text.py \
  --checkpoint ./checkpoints/vq_head/best_*.pt \
  --output ./results/vq_predictions.json

# 评估所有结果
python scripts/eval_text.py --predictions ./results/baseline_predictions.json
python scripts/eval_text.py --predictions ./results/refiner_predictions.json
python scripts/eval_text.py --predictions ./results/vq_predictions.json
```

---

## 🐛 常见问题

### 1. CUDA 内存不足

**错误**: `CUDA out of memory`

**解决方案**:
```bash
# 减小 batch size
python scripts/train_text.py --batch_size 2

# 减小序列长度
python scripts/train_text.py --max_length 128
```

### 2. 数据集下载失败

**错误**: `Connection timeout` 或 `403 Forbidden`

**解决方案**:
```bash
# 使用 HuggingFace 镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 重新下载
cd ../../
python dataset/download_data.py --streaming --max-samples 1000
```

### 3. 模型加载错误

**错误**: `Model path not found`

**解决方案**:
```bash
# 检查模型路径
ls -la models/OmniFlow-v0.5/

# 确保目录结构正确
models/OmniFlow-v0.5/
├── transformer/
├── vae/
├── text_vae/
├── text_encoder_2/
└── vae_tokenizer/
```

### 4. 音频处理错误

**解决方案**:
音频 VAE 会自动下载。如遇问题：
```bash
pip install librosa torchaudio soundfile
```

---

## 📈 性能参考

### 不同 GPU 的训练速度

| GPU | Batch Size | 训练速度 | 内存使用 |
|-----|-----------|---------|---------|
| RTX 3090 (24GB) | 8 | ~200 steps/min | ~18 GB |
| RTX 4090 (24GB) | 16 | ~350 steps/min | ~22 GB |
| A100 (40GB) | 32 | ~600 steps/min | ~35 GB |
| V100 (16GB) | 4 | ~150 steps/min | ~14 GB |

### 实验结果对比（预期）

| 配置 | 参数量 | 训练时间 | BLEU | ROUGE-L | METEOR |
|------|--------|---------|------|---------|--------|
| Baseline | 低 | 1x | 基准 | 基准 | 基准 |
| +Refiner | 中 | 1.1x | ↑ | ↑ | ↑ |
| VQ Head | 中 | 1.2x | ↑ | ↑ | ↑ |
| VQ+Refiner | 高 | 1.3x | ↑↑ | ↑↑ | ↑↑ |

---

## 💡 提示

1. **快速验证**: 先用 streaming 模式下载少量数据测试流程
2. **监控训练**: 关注 PPL 和 Accuracy 的收敛情况
3. **GPU 内存**: 不足时减小 `batch_size` 或 `max_length`
4. **公平对比**: 确保所有实验使用相同数据集和评估集

---

## 📚 相关文档

- [训练脚本](scripts/train_text.py) - 完整训练代码
- [模型定义](omniflow/models/text_decoder_head.py) - TextDecoderHead 和 LatentRefiner
- [数据配置](config/data_config.json) - 数据集配置

---

**祝训练顺利！** 🎉
