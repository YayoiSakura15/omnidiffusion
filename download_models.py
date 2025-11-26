#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OmniFlow 模型下载脚本

自动下载 OmniFlow 所需的所有预训练模型到 jiangqf/models/ 目录
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


# 定义所有需要下载的模型
MODELS = {
    # CLIP 文本编码器 L
    "clip_vit_l": {
        "repo_id": "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        "local_dir": "CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
        "description": "CLIP ViT-L/14 Text & Vision Encoder",
    },
    # T5 文本编码器
    "t5_large": {
        "repo_id": "google/flan-t5-large",
        "local_dir": "flan-t5-large",
        "description": "T5 Large Text Encoder",
    },
    # LanguageBind 音频编码器
    "languagebind_audio": {
        "repo_id": "LanguageBind/LanguageBind_Audio_FT",
        "local_dir": "LanguageBind_Audio_FT",
        "description": "LanguageBind Audio Encoder",
    },
    # OmniFlow 主模型 (需要用户提供实际的模型ID)
    # 如果有公开的 OmniFlow 模型，请修改下面的 repo_id
    "omniflow": {
        "repo_id": "OmniFlow/OmniFlow-v0.5",  # 示例，需要根据实际情况修改
        "local_dir": "OmniFlow-v0.5",
        "description": "OmniFlow Main Model (Transformer + VAE + Text VAE)",
        "optional": True,  # 标记为可选，因为可能需要手动下载
    },
}


def download_model(repo_id, local_dir, description, skip_existing=True):
    """下载单个模型"""
    print(f"\n{'='*70}")
    print(f"📦 {description}")
    print(f"   HuggingFace: {repo_id}")
    print(f"   本地路径: {local_dir}")
    print('='*70)

    if skip_existing and os.path.exists(local_dir):
        print(f"✅ 模型已存在，跳过下载")
        return True

    try:
        print(f"⏳ 开始下载...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✅ 下载完成: {local_dir}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="下载 OmniFlow 所需的预训练模型"
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default="jiangqf/OmniFlows/models",
        help="模型保存根目录 (默认: jiangqf/OmniFlows/models)",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="指定要下载的模型 (clip_vit_l, t5_large, languagebind_audio, omniflow)，不指定则下载全部",
    )

    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="跳过已存在的模型 (默认开启)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载所有模型",
    )

    args = parser.parse_args()

    # 创建模型目录
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(models_dir)

    print("\n" + "="*70)
    print("🚀 OmniFlow 模型下载工具".center(70))
    print("="*70)
    print(f"\n📁 模型保存目录: {models_dir.absolute()}")

    # 确定要下载的模型
    if args.models is None:
        selected_models = list(MODELS.keys())
    else:
        selected_models = args.models

    print(f"📋 将下载以下模型: {', '.join(selected_models)}\n")

    # 下载模型
    skip_existing = args.skip_existing and not args.force
    success_count = 0
    failed_models = []

    for model_name in selected_models:
        if model_name not in MODELS:
            print(f"⚠️  未知模型: {model_name}，跳过")
            continue

        model_info = MODELS[model_name]

        # 处理可选模型
        if model_info.get("optional", False):
            print(f"\n⚠️  {model_info['description']} 是可选模型")
            print(f"   如果您有访问权限，请确认 repo_id: {model_info['repo_id']}")
            user_input = input("   是否继续下载此模型? (y/N): ")
            if user_input.lower() != 'y':
                print("   ⏭️  跳过此模型")
                continue

        success = download_model(
            repo_id=model_info["repo_id"],
            local_dir=model_info["local_dir"],
            description=model_info["description"],
            skip_existing=skip_existing,
        )

        if success:
            success_count += 1
        else:
            failed_models.append(model_name)

    # 总结
    print("\n" + "="*70)
    print("📊 下载总结".center(70))
    print("="*70)
    print(f"✅ 成功: {success_count} 个模型")
    if failed_models:
        print(f"❌ 失败: {len(failed_models)} 个模型 ({', '.join(failed_models)})")
    print("="*70)

    print("\n💡 提示:")
    print("  1. 如果您需要完整的 OmniFlow 模型，请手动下载或提供正确的 repo_id")
    print("  2. 模型文件可能很大 (数 GB)，请确保有足够的磁盘空间和网络带宽")
    print("  3. 下载中断可以重新运行此脚本，会自动断点续传")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
