#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用文本预测脚本（用于评估）

给定：
  - OmniFlow 预训练模型目录 (--model_path)
  - （可选）训练好的 text decoder head checkpoint (--checkpoint)
  - 数据配置 JSON (--data_config) + 选定的 split (--split)
输出：
  - 一个 JSON 文件 (--output_json)，每条样本包含：
      {
        "id": int,
        "reference": str,
        "prediction": str
      }

说明：
  1. 脚本不依赖具体数据集名，只要 data_config[split]
     里用的是 create_dataloader_from_config 能加载的配置即可。
  2. 当前仅做“文本重构”评估（t2t 风格），不显式利用图像/音频条件，
     但可以对 caption / subtitle 一类文本直接评估。
  3. --checkpoint 现在是可选的：
      - 不给 --checkpoint 且不启用 --use_text_decoder_head：
            使用模型原本的 text_output（原始 OmniFlow 文本头）
      - 启用 --use_text_decoder_head 但不提供 checkpoint：
            使用随机初始化的 decoder head（会打印 WARN）
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

# 把项目根目录加入 sys.path，假设脚本放在 OmniFlows/scripts/ 下
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from omniflow.pipelines.omniflow_pipeline import OmniFlowPipeline
from omniflow.data.text_dataset import create_dataloader_from_config


@torch.no_grad()
def run_inference(
    pipeline: OmniFlowPipeline,
    dataloader,
    device: str,
    max_batches: int = None,
):
    """
    对 dataloader 中的样本做前向，生成预测文本。

    当前策略（与 train_text.py 中 evaluate 类似）：
      - 使用 text_vae.encode 得到文本 latent
      - pad 到 joint_attention_dim
      - hidden_states 使用全 0 卷积 latent（不依赖图像）
      - encoder_hidden_states 使用 text latent
      - timestep = 0（不加噪声）
      - 从 logits 里 argmax 得到预测 token，再用 tokenizer 解码成字符串
    """
    pipeline.transformer.eval()

    tokenizer = pipeline.text_vae_tokenizer
    dtype = pipeline.transformer.dtype

    results = []
    global_id = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inferring")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        texts = batch["text"]  # list[str]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        batch_size = input_ids.size(0)

        # 文本 -> latent
        target_latent = pipeline.text_vae.encode(
            texts,
            input_ids=None,
            tokenizer=pipeline.tokenizer_3,
        ).to(device=device, dtype=dtype)  # [B, L, D_text]

        # pad 到 joint_attention_dim
        pad_size = pipeline.transformer.config.joint_attention_dim - target_latent.shape[-1]
        target_latent_padded = torch.nn.functional.pad(target_latent, (0, pad_size))

        # hidden_states 这里用全 0（不依赖图像/音频）
        hidden_states = torch.zeros(
            batch_size,
            16,
            64,
            64,
            device=device,
            dtype=dtype,
        )

        # pooled_projections 简单置 0
        pooled_projections = torch.zeros(
            batch_size,
            pipeline.transformer.config.pooled_projection_dim,
            device=device,
            dtype=dtype,
        )

        # timestep 全 0（不加噪声）
        t_zero = torch.zeros(batch_size, device=device, dtype=dtype)

        output = pipeline.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=target_latent_padded,
            pooled_projections=pooled_projections,
            timestep=t_zero,
            timestep_text=t_zero,
            timestep_audio=t_zero,
            use_text_output=True,
            decode_text=True,
            targets={"input_ids": input_ids},
            target_prompt_embeds=target_latent_padded,
            split_cond=False,
            return_dict=False,
        )

        logits = output["logits"]  # [B, L, vocab]

        # argmax 取预测 token
        pred_ids = logits.argmax(dim=-1)  # [B, L]

        # 解码为字符串（去掉 special tokens）
        pred_texts = tokenizer.batch_decode(
            pred_ids.cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        ref_texts = tokenizer.batch_decode(
            input_ids.cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for ref, pred in zip(ref_texts, pred_texts):
            results.append(
                {
                    "id": global_id,
                    "reference": ref,
                    "prediction": pred,
                }
            )
            global_id += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate text predictions for evaluation")

    # 模型 / checkpoint
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained OmniFlow model directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to trained text decoder head checkpoint (.pt). "
             "If omitted: use existing text_output or randomly initialized decoder head."
    )

    # decoder head 相关（需要和训练时一致，如果你想用 decoder head）
    parser.add_argument("--use_text_decoder_head", action="store_true",
                        help="Use text decoder head (must match training if checkpoint is provided)")
    parser.add_argument("--use_vq_codebook", action="store_true",
                        help="Use VQ codebook in decoder head (must match training if checkpoint is provided)")
    parser.add_argument("--text_decoder_head_dim", type=int, default=2048,
                        help="Decoder head output dim (must match training if checkpoint is provided)")

    # 数据
    parser.add_argument("--data_config", type=str, required=True,
                        help="Path to data_config.json")
    parser.add_argument("--split", type=str, default="val",
                        help="Which split in data_config to use (e.g. train / val / test)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optionally limit number of batches for quick eval")

    # 输出
    parser.add_argument("--output_json", type=str, required=True,
                        help="Where to save predictions JSON")

    # 其他
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    print("\n===============================")
    print("Generate Text Predictions")
    print("===============================\n")
    print(f"Model path      : {args.model_path}")
    print(f"Checkpoint      : {args.checkpoint if args.checkpoint is not None else '(none)'}")
    print(f"Use decoder head: {args.use_text_decoder_head}")
    print(f"Data config     : {args.data_config}")
    print(f"Split           : {args.split}")
    print(f"Output JSON     : {args.output_json}")
    print(f"Device          : {device}")
    print("===============================\n")

    # 1. 加载模型
    print("Loading OmniFlow pipeline...")
    pipeline = OmniFlowPipeline.load_pretrained(
        args.model_path,
        device=device,
        weight_dtype=torch.bfloat16,
        verbose=True,
    )

    # 2. 配置 decoder head（与 train_text.py 一致）
    if args.use_text_decoder_head:
        print("\nEnabling text decoder head...")
        from omniflow.models.text_decoder_head import (
            TextDecoderHead,
            TextDecoderHeadWithCodebook,
        )

        if args.use_vq_codebook:
            pipeline.transformer.text_decoder_head = TextDecoderHeadWithCodebook(
                input_dim=pipeline.transformer.text_out_dim,
                hidden_dim=4096,
                output_dim=args.text_decoder_head_dim,
            ).to(device=device, dtype=pipeline.transformer.dtype)
        else:
            pipeline.transformer.text_decoder_head = TextDecoderHead(
                input_dim=pipeline.transformer.text_out_dim,
                hidden_dim=4096,
                output_dim=args.text_decoder_head_dim,
            ).to(device=device, dtype=pipeline.transformer.dtype)

        pipeline.transformer.use_text_decoder_head = True
        print(f"✓ Text decoder head initialized (output_dim={args.text_decoder_head_dim})")
    else:
        print("\nNot using text decoder head: will rely on model's existing text_output layer.")

    # 3. （可选）加载 checkpoint
    if args.checkpoint is not None:
        print(f"\nLoading checkpoint from {args.checkpoint} ...")
        ckpt = torch.load(args.checkpoint, map_location=device)

        state_dict = ckpt.get("state_dict", ckpt)
        if pipeline.transformer.text_decoder_head is not None:
            pipeline.transformer.text_decoder_head.load_state_dict(state_dict, strict=True)
            print("✓ Loaded state_dict into text_decoder_head")
        else:
            pipeline.transformer.text_output.load_state_dict(state_dict, strict=True)
            print("✓ Loaded state_dict into text_output")
    else:
        if args.use_text_decoder_head:
            print("[WARN] --use_text_decoder_head is set but no --checkpoint is provided.")
            print("       The decoder head will use RANDOMLY INITIALIZED weights (not fine-tuned).")
        else:
            print("No checkpoint provided: using the pretrained model's original text_output weights.\n")

    # 4. 准备数据
    print("\nLoading data...")
    with open(args.data_config, "r") as f:
        data_cfg = json.load(f)

    if args.split not in data_cfg:
        raise ValueError(f"Split '{args.split}' not found in data_config")

    dataloader = create_dataloader_from_config(
        config=data_cfg[args.split],
        tokenizer=pipeline.text_vae_tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"✓ Num batches in split '{args.split}': {len(dataloader)}")

    # 5. 推理
    print("\nRunning inference...")
    results = run_inference(
        pipeline=pipeline,
        dataloader=dataloader,
        device=device,
        max_batches=args.max_batches,
    )

    # 6. 保存 JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} predictions to: {out_path}")
    print("Done.\n")


if __name__ == "__main__":
    main()
