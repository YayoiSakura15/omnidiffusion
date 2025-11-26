#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的文本解码器训练脚本（多模态条件 → 文本）

使用下载的数据集(WikiText, COCO captions, 以及带 image/audio 的 HF datasets)
训练 text decoder head，冻结 OmniFlow 主干网络，只训练 text_decoder_head 或 text_output。
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random  # 用于采样每个样本的任务

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from omniflow.pipelines.omniflow_pipeline import OmniFlowPipeline
from omniflow.data.text_dataset import create_dataloader_from_config
from omniflow.losses import TextReconstructionLoss, PerplexityMetric, TokenAccuracyMetric


class TextTrainer:
    """文本解码器训练器（支持多模态条件）"""

    def __init__(
        self,
        pipeline: OmniFlowPipeline,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: nn.Module,
        device: str = 'cuda',
        output_dir: str = './checkpoints',
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        max_grad_norm: float = 1.0,
    ):
        self.pipeline = pipeline
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_grad_norm = max_grad_norm

        # 指标
        self.ppl_metric = PerplexityMetric().to(device)
        self.acc_metric = TokenAccuracyMetric().to(device)

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_ppl = float('inf')

        # 避免重复打印 audio 相关 warning
        self._warned_audio = False

    # 为每个样本采样一个“模态任务”
    def _sample_modality_tasks(self, batch):
        """
        为 batch 中每个样本采样一个任务:
        - t2t         : 只用文本条件
        - i2t         : 只用图像条件
        - a2t         : 只用音频条件
        - i+t2t       : 图像 + 文本条件
        - a+t2t       : 音频 + 文本条件
        - i+a2t       : 图像 + 音频条件
        - i+a+t2t     : 图像 + 音频 + 文本条件

        返回:
            use_text_cond  : [B] bool
            use_image_cond : [B] bool
            use_audio_cond : [B] bool
            task_names     : List[str]
        """
        texts = batch["text"]
        batch_size = len(texts)

        images = batch.get("image", None)  # List[Image or None] 或不存在
        audios = batch.get("audio", None)  # List[dict or None] 或不存在

        use_text_cond = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        use_image_cond = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        use_audio_cond = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        task_names = []

        for i in range(batch_size):
            has_image = images is not None and images[i] is not None
            has_audio = audios is not None and audios[i] is not None

            if has_image and has_audio:
                candidates = [
                    "t2t",
                    "i+t2t",
                    "a+t2t",
                    "i+a+t2t",
                    "i2t",
                    "a2t",
                    "i+a2t",
                ]
            elif has_image:
                candidates = ["t2t", "i+t2t", "i2t"]
            elif has_audio:
                candidates = ["t2t", "a+t2t", "a2t"]
            else:
                candidates = ["t2t"]

            task = random.choice(candidates)
            task_names.append(task)

            use_text_cond[i] = ("t" in task)
            use_image_cond[i] = ("i" in task)
            use_audio_cond[i] = ("a" in task)

        return use_text_cond, use_image_cond, use_audio_cond, task_names

    def train_step(self, batch):
        """单步训练 - 支持多种模态组合作为条件，输出文本 loss"""
        self.optimizer.zero_grad()

        device = self.device
        transformer_dtype = self.pipeline.transformer.dtype

        # ---- 0. 采样每个样本的“任务类型” ----
        use_text_cond, use_image_cond, use_audio_cond, task_names = self._sample_modality_tasks(batch)

        # ---- 1. 文本数据 & text_vae latent ----
        texts = batch["text"]                       # List[str]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_size = input_ids.size(0)

        with torch.no_grad():
            target_latent = self.pipeline.text_vae.encode(
                texts,
                input_ids=None,
                tokenizer=self.pipeline.tokenizer_3,
            ).to(device=device, dtype=transformer_dtype)   # [B, L, D_text]

        # 加噪（共享一组 timesteps）
        noise_text = torch.randn_like(target_latent)
        t_indices = torch.randint(
            low=0,
            high=len(self.pipeline.scheduler.timesteps),
            size=(batch_size,),
            device=device,
        )
        scheduler_timesteps = self.pipeline.scheduler.timesteps.to(t_indices.device)
        timesteps = scheduler_timesteps[t_indices]

        noisy_latent = self.pipeline.scheduler.scale_noise(
            sample=target_latent,
            timestep=timesteps,
            noise=noise_text,
        )

        # pad 到 joint_attention_dim
        pad_size = self.pipeline.transformer.config.joint_attention_dim - noisy_latent.shape[-1]
        noisy_latent_padded = torch.nn.functional.pad(noisy_latent, (0, pad_size))
        target_latent_padded = torch.nn.functional.pad(target_latent, (0, pad_size))

        # 对 encoder_hidden_states 应用 text 条件 mask
        encoder_hidden_states = noisy_latent_padded.clone()
        if not use_text_cond.all():
            mask_text = use_text_cond.to(device=device, dtype=encoder_hidden_states.dtype).view(batch_size, 1, 1)
            encoder_hidden_states = encoder_hidden_states * mask_text

        # ---- 2. 图像分支：VAE latent + 噪声 + image 条件 mask ----
        from PIL import Image

        images = batch.get("image", None)  # List[Image or None]
        image_latents = torch.zeros(
            batch_size,
            16,
            64,
            64,
            device=device,
            dtype=transformer_dtype,
        )

        if images is not None:
            proc_imgs = []
            valid_indices = []

            for i, img in enumerate(images):
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    # dataset 如果给的是路径，可以这里 open；当前 HFDatasetWrapper 已经给的是 Image 或 None
                    try:
                        img = Image.open(img).convert("RGB")
                    except Exception:
                        continue

                # 中心裁剪成方形，resize 到 512x512
                w, h = img.size
                min_side = min(w, h)
                left = (w - min_side) / 2
                top = (h - min_side) / 2
                img = img.crop((left, top, left + min_side, top + min_side))
                img = img.resize((512, 512))
                img = img.convert("RGB")

                proc_imgs.append(img)
                valid_indices.append(i)

            if len(proc_imgs) > 0:
                pixel_values = self.pipeline.image_processor.preprocess(proc_imgs).to(
                    device=self.device,
                    dtype=self.pipeline.vae.dtype,
                )
                latents = self.pipeline.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.pipeline.vae.config.scaling_factor
                latents = latents.to(dtype=transformer_dtype)

                for k, idx in enumerate(valid_indices):
                    image_latents[idx] = latents[k]

        # 给 image latent 加噪（用同一组 timesteps）
        noise_img = torch.randn_like(image_latents)
        noisy_image_latents = self.pipeline.scheduler.scale_noise(
            sample=image_latents,
            timestep=timesteps,
            noise=noise_img,
        )

        # 应用 image 条件 mask
        if not use_image_cond.all():
            mask_img = use_image_cond.to(device=device, dtype=noisy_image_latents.dtype).view(batch_size, 1, 1, 1)
            noisy_image_latents = noisy_image_latents * mask_img

        # ---- 3. 音频分支：audio_vae latent + 噪声 + audio 条件 mask ----
        audios = batch.get("audio", None)  # List[dict 或 str 或 None]
        audio_hidden_states = None

        has_audio_in_batch = audios is not None and any(a is not None for a in audios)

        if has_audio_in_batch and use_audio_cond.any():
            audio_vae = getattr(self.pipeline, "audio_vae", None)

            if audio_vae is None:
                # 有音频数据，但 pipeline 里没有 audio_vae
                if not self._warned_audio:
                    print("[WARN] Batch 中存在 audio 条件，但当前 pipeline 未加载 audio_vae，音频条件将被忽略。")
                    self._warned_audio = True
            else:
                import torchaudio

                B = batch_size
                audio_dtype = getattr(audio_vae, "dtype", torch.float32)

                # 统一到 [B, 1, 1024, 64]
                T = 1024
                F = 64
                fbank_batch = torch.zeros(
                    B,
                    T,
                    F,
                    device=device,
                    dtype=audio_dtype,
                )

                for i, a in enumerate(audios):
                    # 当前样本没音频 或 本 step 不用音频条件
                    if a is None or not use_audio_cond[i]:
                        continue

                    arr = None
                    sr = None

                    # 情况 1：HF 的 Audio 特征，dict 里有 array / sampling_rate
                    if isinstance(a, dict):
                        arr = a.get("array", None)
                        sr = a.get("sampling_rate", None)

                        # 有些数据集只给了 path，我们就从 path 读
                        if (arr is None or sr is None) and "path" in a:
                            try:
                                wav_tensor, sr_ = torchaudio.load(a["path"])
                                # [C, T] → mono [T]
                                if wav_tensor.ndim > 1:
                                    wav_tensor = wav_tensor.mean(dim=0)
                                arr = wav_tensor.cpu().numpy()
                                sr = sr_
                            except Exception:
                                arr = None
                                sr = None

                    # 情况 2：直接就是音频文件路径字符串
                    elif isinstance(a, str):
                        try:
                            wav_tensor, sr_ = torchaudio.load(a)
                            if wav_tensor.ndim > 1:
                                wav_tensor = wav_tensor.mean(dim=0)
                            arr = wav_tensor.cpu().numpy()
                            sr = sr_
                        except Exception:
                            arr = None
                            sr = None

                    # 既没 array 也没 sr，就跳过这个样本的 audio
                    if arr is None or sr is None:
                        continue

                    # arr -> waveform tensor
                    wav = torch.tensor(arr, dtype=torch.float32, device=device)
                    if wav.ndim > 1:
                        wav = wav.mean(dim=0)
                    wav = wav.unsqueeze(0)  # [1, num_samples]

                    # 计算 fbank 特征
                    with torch.no_grad():
                        fb = torchaudio.compliance.kaldi.fbank(
                            wav,
                            sample_frequency=float(sr),
                            num_mel_bins=F,
                            use_energy=False,
                            window_type="hanning",
                        )  # [frames, F]

                    # pad / crop 到固定 T 帧
                    if fb.size(0) < T:
                        pad_frames = T - fb.size(0)
                        fb = torch.nn.functional.pad(fb, (0, 0, 0, pad_frames))
                    else:
                        fb = fb[:T]

                    fbank_batch[i] = fb  # [T, F]

                # [B, 1, T, F] 喂给 audio_vae
                fbank_batch = fbank_batch.unsqueeze(1)

                with torch.no_grad():
                    raw_audio_latents = audio_vae.encode(fbank_batch).latent_dist.sample()
                    raw_audio_latents = raw_audio_latents * audio_vae.config.scaling_factor
                    raw_audio_latents = raw_audio_latents.to(device=device, dtype=transformer_dtype)

                # 与 text/image 共用同一组 timesteps 加噪
                noise_audio = torch.randn_like(raw_audio_latents)
                noisy_audio_latents = self.pipeline.scheduler.scale_noise(
                    sample=raw_audio_latents,
                    timestep=timesteps,
                    noise=noise_audio,
                )

                # 按任务 mask 掉不使用音频的样本
                if not use_audio_cond.all():
                    mask_audio = use_audio_cond.to(
                        device=device,
                        dtype=noisy_audio_latents.dtype,
                    ).view(B, 1, 1, 1)
                    noisy_audio_latents = noisy_audio_latents * mask_audio

                audio_hidden_states = noisy_audio_latents


        # ---- 4. pooled_projections：简单编码当前样本用了哪些模态 ----
        pooled_projections = torch.zeros(
            batch_size,
            self.pipeline.transformer.config.pooled_projection_dim,
            device=device,
            dtype=transformer_dtype,
        )

        with torch.no_grad():
            pooled_projections[:, 0] = use_image_cond.float()
            pooled_projections[:, 1] = use_audio_cond.float()
            pooled_projections[:, 2] = use_text_cond.float()

        # ---- 5. 调用 transformer ----
        t_model = timesteps.to(device=device, dtype=transformer_dtype)
        t_audio = t_model.clone()

        output = self.pipeline.transformer(
            hidden_states=noisy_image_latents,
            timestep=t_model,
            timestep_text=t_model,
            timestep_audio=t_audio,
            encoder_hidden_states=encoder_hidden_states,
            audio_hidden_states=audio_hidden_states,
            pooled_projections=pooled_projections,
            use_text_output=True,
            decode_text=True,
            targets={"input_ids": input_ids},
            target_prompt_embeds=target_latent_padded,
            split_cond=False,  # 训练脚本里用 False，与原 omni-flow 训练脚本一致
            return_dict=False,
        )

        pred_latent = output["model_pred_text"]  # [B, L, D_pred]
        logits = output["logits"]               # [B, L, vocab]
        vq_loss = output.get("vq_loss", None)

        # ---- 5.5. 应用 LatentRefiner（如果有）----
        if hasattr(self.pipeline.transformer, 'latent_refiner') and self.pipeline.transformer.latent_refiner is not None:
            pred_latent = self.pipeline.transformer.latent_refiner(pred_latent)

        # ---- 6. 计算损失 ----
        if pred_latent.shape[-1] != target_latent.shape[-1]:
            target_latent_expanded = torch.zeros_like(pred_latent)
            target_latent_expanded[..., :target_latent.shape[-1]] = target_latent
            target_latent_for_loss = target_latent_expanded
        else:
            target_latent_for_loss = target_latent

        loss_dict = self.loss_fn(
            logits=logits,
            target_ids=input_ids,
            pred_latent=pred_latent,
            target_latent=target_latent_for_loss,
            vq_loss=vq_loss,
            attention_mask=attention_mask,
        )

        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        # 只剪裁 & 更新 text decoder head / latent_refiner / text_output
        if self.pipeline.transformer.text_decoder_head is not None:
            torch.nn.utils.clip_grad_norm_(
                self.pipeline.transformer.text_decoder_head.parameters(),
                self.max_grad_norm,
            )
            # 剪裁 LatentRefiner 梯度（如果有）
            if hasattr(self.pipeline.transformer, 'latent_refiner') and self.pipeline.transformer.latent_refiner is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.pipeline.transformer.latent_refiner.parameters(),
                    self.max_grad_norm,
                )
        else:
            torch.nn.utils.clip_grad_norm_(
                self.pipeline.transformer.text_output.parameters(),
                self.max_grad_norm,
            )

        self.optimizer.step()
        self.scheduler.step()

        # ---- 7. 指标 ----
        with torch.no_grad():
            perplexity = self.ppl_metric(logits, input_ids, attention_mask)
            accuracy = self.acc_metric(logits, input_ids, attention_mask)

        return {
            "total_loss": total_loss.item(),
            "ce_loss": loss_dict["ce_loss"].item(),
            "mse_loss": loss_dict["mse_loss"].item(),
            "perplexity": perplexity.item(),
            "accuracy": accuracy.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def evaluate(self):
        """验证（保持简单：只用 text 条件）"""
        self.pipeline.transformer.eval()

        total_ppl = 0
        total_acc = 0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            target_latent = self.pipeline.text_vae.encode(
                batch['text'],
                input_ids=None,
                tokenizer=self.pipeline.tokenizer_3
            ).to(device=self.device, dtype=self.pipeline.transformer.dtype)

            pad_size = self.pipeline.transformer.config.joint_attention_dim - target_latent.shape[-1]
            target_latent_padded = torch.nn.functional.pad(target_latent, (0, pad_size))

            batch_size = target_latent.shape[0]
            pooled_projections = torch.zeros(
                batch_size,
                self.pipeline.transformer.config.pooled_projection_dim,
                device=self.device,
                dtype=self.pipeline.transformer.dtype
            )

            output = self.pipeline.transformer(
                hidden_states=torch.zeros(
                    batch_size, 16, 64, 64,
                    device=self.device,
                    dtype=self.pipeline.transformer.dtype
                ),
                encoder_hidden_states=target_latent_padded,
                pooled_projections=pooled_projections,
                timestep=torch.zeros(batch_size, device=self.device, dtype=self.pipeline.transformer.dtype),
                timestep_text=torch.zeros(batch_size, device=self.device, dtype=self.pipeline.transformer.dtype),
                timestep_audio=torch.zeros(batch_size, device=self.device, dtype=self.pipeline.transformer.dtype) + 1000,
                use_text_output=True,
                decode_text=True,
                targets={'input_ids': input_ids},
                target_prompt_embeds=target_latent_padded,
                split_cond=False,  # 与训练保持一致
                return_dict=False,
            )

            logits = output['logits']

            ppl = self.ppl_metric(logits, input_ids, attention_mask)
            acc = self.acc_metric(logits, input_ids, attention_mask)

            total_ppl += ppl.item()
            total_acc += acc.item()
            num_batches += 1

            if num_batches >= 50:  # 限制验证批次数
                break

        avg_ppl = total_ppl / num_batches
        avg_acc = total_acc / num_batches

        self.pipeline.transformer.train()

        return {
            'perplexity': avg_ppl,
            'accuracy': avg_acc,
        }

    def save_checkpoint(self, name='checkpoint'):
        """保存检查点（只存解码头或 text_output，以及 LatentRefiner）"""
        checkpoint_path = self.output_dir / f'{name}_step{self.global_step}.pt'

        checkpoint_dict = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_ppl': self.best_val_ppl,
        }

        if self.pipeline.transformer.text_decoder_head is not None:
            checkpoint_dict['text_decoder_head'] = self.pipeline.transformer.text_decoder_head.state_dict()
            # 保存 LatentRefiner（如果有）
            if hasattr(self.pipeline.transformer, 'latent_refiner') and self.pipeline.transformer.latent_refiner is not None:
                checkpoint_dict['latent_refiner'] = self.pipeline.transformer.latent_refiner.state_dict()
        else:
            checkpoint_dict['state_dict'] = self.pipeline.transformer.text_output.state_dict()

        torch.save(checkpoint_dict, checkpoint_path)

        print(f"✓ Saved checkpoint to {checkpoint_path}")

    def train(self, num_epochs: int):
        """主训练循环"""
        print(f"\n{'='*60}")
        print("Starting Text Decoder Training")
        print(f"{'='*60}\n")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training batches per epoch: {len(self.train_dataloader)}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            self.pipeline.transformer.train()

            for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")):
                metrics = self.train_step(batch)

                self.global_step += 1

                if self.global_step % self.log_interval == 0:
                    log_str = f"Step {self.global_step} | "
                    log_str += f"Loss: {metrics['total_loss']:.4f} | "
                    log_str += f"CE: {metrics['ce_loss']:.4f} | "
                    log_str += f"MSE: {metrics['mse_loss']:.4f} | "
                    log_str += f"PPL: {metrics['perplexity']:.2f} | "
                    log_str += f"Acc: {metrics['accuracy']:.4f} | "
                    log_str += f"LR: {metrics['lr']:.2e}"
                    print(log_str)

                if self.global_step % self.eval_interval == 0:
                    val_metrics = self.evaluate()
                    print(f"\n[Validation] PPL: {val_metrics['perplexity']:.2f}, "
                          f"Acc: {val_metrics['accuracy']:.4f}\n")

                    if val_metrics['perplexity'] < self.best_val_ppl:
                        self.best_val_ppl = val_metrics['perplexity']
                        self.save_checkpoint('best')

                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint('latest')

        self.save_checkpoint('final')

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best Validation Perplexity: {self.best_val_ppl:.2f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train text decoder head on downloaded datasets')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained OmniFlow model')
    parser.add_argument('--use_text_decoder_head', action='store_true',
                        help='Use text decoder head')
    parser.add_argument('--use_vq_codebook', action='store_true',
                        help='Use VQ codebook in decoder head')
    parser.add_argument('--text_decoder_head_dim', type=int, default=2048,
                        help='Decoder head output dimension')

    # LatentRefiner 参数
    parser.add_argument('--use_latent_refiner', action='store_true',
                        help='Use LatentRefiner after text decoder head')
    parser.add_argument('--latent_refiner_hidden_dim', type=int, default=256,
                        help='LatentRefiner hidden dimension')
    parser.add_argument('--latent_refiner_layers', type=int, default=2,
                        help='Number of LatentRefiner MLP layers')

    # 数据参数
    parser.add_argument('--data_config', type=str, default='../config/data_config.json',
                        help='Path to data configuration JSON')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Warmup steps')

    # 损失权重
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='Cross entropy loss weight')
    parser.add_argument('--mse_weight', type=float, default=0.1,
                        help='MSE loss weight')
    parser.add_argument('--contrastive_weight', type=float, default=0.05,
                        help='Contrastive loss weight')
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Use contrastive loss')

    # 其他
    parser.add_argument('--output_dir', type=str, default='./checkpoints/text_decoder',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Evaluate every N steps')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Save checkpoint every N steps')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("\n" + "="*60)
    print("OmniFlow Text Decoder Training")
    print("="*60 + "\n")

    # 1. 加载模型
    print("Loading OmniFlow pipeline...")
    pipeline = OmniFlowPipeline.load_pretrained(
        args.model_path,
        device=args.device,
        weight_dtype=torch.bfloat16,
        verbose=True,
    )

    # 2. 添加 decoder head（如需要）
    if args.use_text_decoder_head:
        print("\nEnabling text decoder head...")
        from omniflow.models.text_decoder_head import (
            TextDecoderHead,
            TextDecoderHeadWithCodebook,
            LatentRefiner
        )

        if args.use_vq_codebook:
            pipeline.transformer.text_decoder_head = TextDecoderHeadWithCodebook(
                input_dim=pipeline.transformer.text_out_dim,
                hidden_dim=4096,
                output_dim=args.text_decoder_head_dim,
            ).to(device=args.device, dtype=pipeline.transformer.dtype)
            print(f"✓ Text decoder head with VQ initialized (output_dim={args.text_decoder_head_dim})")
        else:
            pipeline.transformer.text_decoder_head = TextDecoderHead(
                input_dim=pipeline.transformer.text_out_dim,
                hidden_dim=4096,
                output_dim=args.text_decoder_head_dim,
            ).to(device=args.device, dtype=pipeline.transformer.dtype)
            print(f"✓ Text decoder head initialized (output_dim={args.text_decoder_head_dim})")

        pipeline.transformer.use_text_decoder_head = True

        # 添加 LatentRefiner（如需要）
        if args.use_latent_refiner:
            print("\nEnabling LatentRefiner...")
            pipeline.transformer.latent_refiner = LatentRefiner(
                latent_dim=args.text_decoder_head_dim,
                hidden_dim=args.latent_refiner_hidden_dim,
                num_layers=args.latent_refiner_layers,
            ).to(device=args.device, dtype=pipeline.transformer.dtype)
            print(f"✓ LatentRefiner initialized (hidden_dim={args.latent_refiner_hidden_dim}, layers={args.latent_refiner_layers})")
        else:
            pipeline.transformer.latent_refiner = None

    # 3. 冻结预训练组件
    print("\nFreezing pretrained components...")

    for param in pipeline.vae.parameters():
        param.requires_grad = False
    print("✓ VAE frozen")

    for param in pipeline.text_vae.parameters():
        param.requires_grad = False
    print("✓ Text VAE frozen")

    for param in pipeline.transformer.parameters():
        param.requires_grad = False
    print("✓ Transformer backbone frozen")

    # 只解冻 decoder head / latent_refiner / text_output
    trainable_params = 0

    if pipeline.transformer.text_decoder_head is not None:
        for param in pipeline.transformer.text_decoder_head.parameters():
            param.requires_grad = True
        head_params = sum(p.numel() for p in pipeline.transformer.text_decoder_head.parameters())
        trainable_params += head_params
        print(f"✓ Text decoder head unfrozen ({head_params:,} params)")

        # 解冻 LatentRefiner（如果有）
        if hasattr(pipeline.transformer, 'latent_refiner') and pipeline.transformer.latent_refiner is not None:
            for param in pipeline.transformer.latent_refiner.parameters():
                param.requires_grad = True
            refiner_params = sum(p.numel() for p in pipeline.transformer.latent_refiner.parameters())
            trainable_params += refiner_params
            print(f"✓ LatentRefiner unfrozen ({refiner_params:,} params)")
    else:
        for param in pipeline.transformer.text_output.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in pipeline.transformer.text_output.parameters())
        print("✓ Text output layer unfrozen")

    print(f"✓ Total trainable parameters: {trainable_params:,}")

    # 4. 准备数据
    print("\nLoading data...")
    with open(args.data_config, 'r') as f:
        data_config = json.load(f)

    train_dataloader = create_dataloader_from_config(
        config=data_config['train'],
        tokenizer=pipeline.text_vae_tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_dataloader = create_dataloader_from_config(
        config=data_config['val'],
        tokenizer=pipeline.text_vae_tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"✓ Train batches: {len(train_dataloader)}")
    print(f"✓ Val batches: {len(val_dataloader)}")

    # 5. 优化器 & scheduler
    print("\nSetting up optimizer...")

    # 收集需要优化的参数
    trainable_modules = []
    if pipeline.transformer.text_decoder_head is not None:
        trainable_modules.append(pipeline.transformer.text_decoder_head)
        # 添加 LatentRefiner 参数（如果有）
        if hasattr(pipeline.transformer, 'latent_refiner') and pipeline.transformer.latent_refiner is not None:
            trainable_modules.append(pipeline.transformer.latent_refiner)
    else:
        trainable_modules.append(pipeline.transformer.text_output)

    # 将所有可训练模块的参数合并
    trainable_params_list = []
    for module in trainable_modules:
        trainable_params_list.extend(list(module.parameters()))

    optimizer = AdamW(
        trainable_params_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_dataloader) * args.num_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - args.warmup_steps, 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

    # 6. 损失函数
    print("\nSetting up loss function...")
    loss_fn = TextReconstructionLoss(
        ce_weight=args.ce_weight,
        mse_weight=args.mse_weight,
        contrastive_weight=args.contrastive_weight,
        use_contrastive=args.use_contrastive,
    ).to(args.device)

    # 7. Trainer
    trainer = TextTrainer(
        pipeline=pipeline,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=args.device,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )

    trainer.train(num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()
