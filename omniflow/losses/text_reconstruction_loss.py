"""
Text Reconstruction Loss Module
组合多种损失函数来优化文本解码器头部
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class TextReconstructionLoss(nn.Module):
    """
    文本重建损失

    组合以下损失函数:
    1. Token-level Cross Entropy Loss (重建损失)
    2. Latent Space MSE Loss (潜在空间对齐)
    3. Contrastive Loss (可选，确保相似文本的latent接近)
    4. VQ Loss (如果使用VQ Codebook)

    Args:
        ce_weight (float): CE损失权重
        mse_weight (float): MSE损失权重
        contrastive_weight (float): 对比损失权重
        vq_weight (float): VQ损失权重
        label_smoothing (float): 标签平滑系数
        use_contrastive (bool): 是否使用对比损失
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        mse_weight: float = 0.1,
        contrastive_weight: float = 0.05,
        vq_weight: float = 1.0,
        label_smoothing: float = 0.0,
        use_contrastive: bool = False,
        temperature: float = 0.07,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.mse_weight = mse_weight
        self.contrastive_weight = contrastive_weight
        self.vq_weight = vq_weight
        self.use_contrastive = use_contrastive
        self.temperature = temperature

        # Token重建损失
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction='mean'
        )

        # 潜在空间对齐损失
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        vq_loss: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [batch_size, seq_len, vocab_size] 预测的logits
            target_ids: [batch_size, seq_len] 目标token ids
            pred_latent: [batch_size, L_latent, latent_dim] 预测的潜在向量
            target_latent: [batch_size, L_latent, latent_dim] 目标潜在向量
            vq_loss: VQ损失（如果使用）
            attention_mask: [batch_size, seq_len_tok] 注意力掩码

        Returns:
            loss_dict: 包含各种损失的字典
        """
        loss_dict = {}

        # 1. Token重建损失 (Cross Entropy)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        ce_loss = self.ce_loss(logits_flat, targets_flat)
        loss_dict['ce_loss'] = ce_loss

        # 2. 潜在空间对齐损失 (MSE)
        #
        # 注意：这里的 pred_latent / target_latent 是 text VAE 的 latent 序列，
        # 通常长度 L_latent != token 序列长度。
        # 所以不能直接用 token 的 attention_mask 去乘，否则会 32 vs 256 这种 mismatch。
        #
        # 处理策略：
        # - 如果长度刚好相等：沿用原来的 masked MSE 逻辑；
        # - 否则：对整个 latent 序列做 MSE，不使用 token mask。
        if self.mse_weight > 0:
            if (
                attention_mask is not None
                and attention_mask.shape[1] == pred_latent.shape[1]  # <<< CHANGED: 只有长度匹配才用 token mask
            ):
                mask = attention_mask.unsqueeze(-1).to(dtype=pred_latent.dtype, device=pred_latent.device)  # [B, L, 1]
                masked_pred = pred_latent * mask
                masked_target = target_latent * mask
                mse_loss = self.mse_loss(masked_pred, masked_target)
            else:
                # 长度不匹配：latent 自己用全 1 mask，避免 32 vs 256 的尺寸冲突
                mse_loss = self.mse_loss(pred_latent, target_latent)  # <<< CHANGED: 不再用 token mask
        else:
            mse_loss = logits.new_tensor(0.0)

        loss_dict['mse_loss'] = mse_loss

        # 3. 对比损失 (可选)
        if self.use_contrastive:
            contrastive_loss = self._compute_contrastive_loss(
                pred_latent, target_latent, attention_mask
            )
            loss_dict['contrastive_loss'] = contrastive_loss
        else:
            loss_dict['contrastive_loss'] = torch.tensor(0.0, device=logits.device)

        # 4. VQ损失 (如果使用VQ Codebook)
        if vq_loss is not None:
            loss_dict['vq_loss'] = vq_loss
        else:
            loss_dict['vq_loss'] = torch.tensor(0.0, device=logits.device)

        # 总损失
        total_loss = (
            self.ce_weight * ce_loss +
            self.mse_weight * mse_loss +
            self.contrastive_weight * loss_dict['contrastive_loss'] +
            self.vq_weight * loss_dict['vq_loss']
        )

        loss_dict['total_loss'] = total_loss

        return loss_dict

    def _compute_contrastive_loss(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算对比损失（句子级别）

        Args:
            pred_latent: [B, L_latent, D]
            target_latent: [B, L_latent, D]
            attention_mask: [B, L_tok]

        Returns:
            contrastive_loss: scalar
        """
        # 对序列进行pooling得到句子级表示
        if (
            attention_mask is not None
            and attention_mask.shape[1] == pred_latent.shape[1]   # <<< CHANGED: 只有长度一致才用 mask
        ):
            # Masked average pooling
            mask = attention_mask.unsqueeze(-1).to(dtype=pred_latent.dtype, device=pred_latent.device)  # [B, L, 1]
            pred_pooled = (pred_latent * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            target_pooled = (target_latent * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            # Mean pooling（长度不匹配时走这里）
            pred_pooled = pred_latent.mean(dim=1)    # [B, D]
            target_pooled = target_latent.mean(dim=1)  # [B, D]

        # L2 归一化
        pred_pooled = F.normalize(pred_pooled, p=2, dim=-1)
        target_pooled = F.normalize(target_pooled, p=2, dim=-1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(pred_pooled, target_pooled.T) / self.temperature
        # [B, B]

        # InfoNCE损失
        batch_size = similarity_matrix.shape[0]
        labels = torch.arange(batch_size, device=similarity_matrix.device)

        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


class PerplexityMetric(nn.Module):
    """
    困惑度指标，用于评估文本生成质量
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, L, V]
            target_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            perplexity: scalar
        """
        # 计算每个token的log probability
        log_probs = F.log_softmax(logits, dim=-1)  # [B, L, V]

        # 收集目标token的log prob
        batch_size, seq_len, vocab_size = logits.shape
        target_log_probs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [B, L]

        # 应用mask
        if attention_mask is not None:
            target_log_probs = target_log_probs * attention_mask
            num_tokens = attention_mask.sum()
        else:
            num_tokens = batch_size * seq_len

        # 计算平均negative log likelihood
        avg_nll = -target_log_probs.sum() / num_tokens

        # 困惑度 = exp(NLL)
        perplexity = torch.exp(avg_nll)

        return perplexity


class TokenAccuracyMetric(nn.Module):
    """
    Token级别的准确率
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, L, V]
            target_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            accuracy: scalar (0-1)
        """
        # 预测的token
        pred_ids = logits.argmax(dim=-1)  # [B, L]

        # 计算准确匹配
        correct = (pred_ids == target_ids).float()

        # 应用mask
        if attention_mask is not None:
            correct = correct * attention_mask
            num_tokens = attention_mask.sum()
        else:
            num_tokens = correct.numel()

        accuracy = correct.sum() / num_tokens

        return accuracy


if __name__ == "__main__":
    # 测试代码
    batch_size = 4
    seq_len = 128
    vocab_size = 50000
    latent_dim = 2048

    # 创建模拟数据
    logits = torch.randn(batch_size, seq_len, vocab_size)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    pred_latent = torch.randn(batch_size, 32, latent_dim)      # <<< 注意这里 latent 长度改成 32 也不会再炸
    target_latent = torch.randn(batch_size, 32, latent_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, seq_len//2:] = 0  # 模拟padding

    # 测试损失函数
    print("Testing TextReconstructionLoss...")
    loss_fn = TextReconstructionLoss(
        ce_weight=1.0,
        mse_weight=0.1,
        contrastive_weight=0.05,
        use_contrastive=True
    )

    loss_dict = loss_fn(
        logits=logits,
        target_ids=target_ids,
        pred_latent=pred_latent,
        target_latent=target_latent,
        attention_mask=attention_mask
    )

    print("\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

    # 测试困惑度
    print("\nTesting PerplexityMetric...")
    ppl_metric = PerplexityMetric()
    perplexity = ppl_metric(logits, target_ids, attention_mask)
    print(f"  Perplexity: {perplexity.item():.2f}")

    # 测试准确率
    print("\nTesting TokenAccuracyMetric...")
    acc_metric = TokenAccuracyMetric()
    accuracy = acc_metric(logits, target_ids, attention_mask)
    print(f"  Token Accuracy: {accuracy.item():.4f}")
