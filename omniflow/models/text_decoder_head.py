"""
Text Decoder Head Module
用于在文本解码前对潜在向量进行上采样和非线性变换，减少连续空间到离散空间的映射误差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TextDecoderHead(nn.Module):
    """
    文本解码器头部模块

    通过上采样和非线性变换将潜在向量映射到更高维度空间，
    使得不同token在连续空间中有更大的"安全距离"

    Args:
        input_dim (int): 输入潜在向量维度，默认 1536
        hidden_dim (int): 隐藏层维度，默认 4096
        output_dim (int): 输出维度，默认 2048
        num_layers (int): MLP 层数，默认 2
        dropout (float): Dropout 比例，默认 0.1
        activation (str): 激活函数类型，默认 'gelu'
        use_residual (bool): 是否使用残差连接，默认 True
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 4096,
        output_dim: int = 2048,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_residual: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual and (input_dim == output_dim)

        # 选择激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 构建多层MLP
        layers = []

        # 第一层: input_dim -> hidden_dim
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation,
            nn.Dropout(dropout),
        ])

        # 中间层: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ])

        # 最后一层: hidden_dim -> output_dim
        layers.extend([
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        ])

        self.upsampler = nn.Sequential(*layers)

        # 如果使用残差连接且维度不匹配，需要投影层
        if self.use_residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [batch_size, seq_len, input_dim]

        Returns:
            upsampled_latents: [batch_size, seq_len, output_dim]
        """
        # 上采样和非线性变换
        output = self.upsampler(latents)

        # 残差连接
        if self.use_residual:
            if self.residual_proj is not None:
                latents = self.residual_proj(latents)
            output = output + latents

        return output


class TextDecoderHeadWithCodebook(nn.Module):
    """
    带有 Codebook 的文本解码器头部（可选）

    参考 VQ-VAE，在上采样后加入离散化 codebook，
    进一步减少连续-离散转换的误差

    Args:
        input_dim (int): 输入维度
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度
        num_embeddings (int): Codebook 大小
        commitment_cost (float): VQ 损失中的承诺成本系数
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 4096,
        output_dim: int = 2048,
        num_embeddings: int = 8192,
        commitment_cost: float = 0.25,
        **kwargs
    ):
        super().__init__()

        # 上采样模块
        self.upsampler = TextDecoderHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )

        # Codebook (VQ layer)
        self.num_embeddings = num_embeddings
        self.embedding_dim = output_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, output_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, latents: torch.Tensor, return_loss: bool = False):
        """
        Args:
            latents: [B, L, input_dim]
            return_loss: 是否返回 VQ 损失

        Returns:
            quantized: [B, L, output_dim] 量化后的向量
            vq_loss: VQ 损失（如果 return_loss=True）
        """
        # 上采样
        z = self.upsampler(latents)  # [B, L, output_dim]

        # 展平以便查找最近邻
        z_flattened = z.reshape(-1, self.embedding_dim)  # [B*L, output_dim]

        # 计算距离并找到最近的 codebook entry
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )  # [B*L, num_embeddings]

        encoding_indices = torch.argmin(distances, dim=1)  # [B*L]
        quantized = self.embedding(encoding_indices).view(z.shape)  # [B, L, output_dim]

        # VQ 损失
        if return_loss:
            # Commitment loss
            e_latent_loss = F.mse_loss(quantized.detach(), z)
            q_latent_loss = F.mse_loss(quantized, z.detach())
            vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

            # Straight-through estimator
            quantized = z + (quantized - z).detach()

            return quantized, vq_loss
        else:
            # Straight-through estimator
            quantized = z + (quantized - z).detach()
            return quantized


class AdaptiveTextDecoderHead(nn.Module):
    """
    自适应文本解码器头部

    根据 timestep 动态调整上采样强度，
    在 diffusion 过程的不同阶段使用不同的上采样策略
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 4096,
        output_dim: int = 2048,
        time_embed_dim: int = 256,
        **kwargs
    ):
        super().__init__()

        self.base_upsampler = TextDecoderHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )

        # Time embedding for adaptive control
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        latents: torch.Tensor,
        timestep_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, L, input_dim]
            timestep_embed: [B, time_embed_dim] or None
        """
        output = self.base_upsampler(latents)

        # 如果提供时间嵌入，进行自适应调制
        if timestep_embed is not None:
            scale = self.time_mlp(timestep_embed)  # [B, output_dim]
            scale = scale.unsqueeze(1)  # [B, 1, output_dim]
            output = output * (1 + scale)

        return output


class LatentRefiner(nn.Module):
    """
    Latent Refiner 模块

    在 text_decoder_head 输出的低维 latent (如 64-d) 上额外加一个小型 MLP，
    进一步优化 latent 表征，提升文本解码质量。

    Args:
        latent_dim (int): latent 维度，应与 text_decoder_head 的 output_dim 一致
        hidden_dim (int): 隐藏层维度，默认 256
        num_layers (int): MLP 层数，默认 2
        dropout (float): Dropout 比例，默认 0.1
        activation (str): 激活函数类型，默认 'gelu'
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # 选择激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 构建 MLP
        layers = []

        # 第一层: latent_dim -> hidden_dim
        layers.extend([
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation,
            nn.Dropout(dropout),
        ])

        # 中间层: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ])

        # 最后一层: hidden_dim -> latent_dim (回到原维度)
        layers.extend([
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        ])

        self.refiner = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [batch_size, seq_len, latent_dim]

        Returns:
            refined_latents: [batch_size, seq_len, latent_dim]
        """
        # Refine with residual connection
        refined = self.refiner(latents)
        return latents + refined  # 残差连接


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    seq_len = 128
    input_dim = 1536

    # 测试基础版本
    print("Testing TextDecoderHead...")
    model = TextDecoderHead(input_dim=1536, hidden_dim=4096, output_dim=2048)
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试 Codebook 版本
    print("\nTesting TextDecoderHeadWithCodebook...")
    model_vq = TextDecoderHeadWithCodebook(
        input_dim=1536,
        hidden_dim=4096,
        output_dim=2048,
        num_embeddings=8192
    )
    y_vq, vq_loss = model_vq(x, return_loss=True)
    print(f"Input shape: {x.shape}, Output shape: {y_vq.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model_vq.parameters()) / 1e6:.2f}M")

    # 测试自适应版本
    print("\nTesting AdaptiveTextDecoderHead...")
    model_adaptive = AdaptiveTextDecoderHead(
        input_dim=1536,
        hidden_dim=4096,
        output_dim=2048,
        time_embed_dim=256
    )
    time_embed = torch.randn(batch_size, 256)
    y_adaptive = model_adaptive(x, time_embed)
    print(f"Input shape: {x.shape}, Output shape: {y_adaptive.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_adaptive.parameters()) / 1e6:.2f}M")

    # 测试 LatentRefiner
    print("\nTesting LatentRefiner...")
    model_refiner = LatentRefiner(latent_dim=64, hidden_dim=256, num_layers=2)
    x_latent = torch.randn(batch_size, seq_len, 64)
    y_refined = model_refiner(x_latent)
    print(f"Input shape: {x_latent.shape}, Output shape: {y_refined.shape}")
    print(f"Parameters: {sum(p.numel() for p in model_refiner.parameters()) / 1e3:.2f}K")
