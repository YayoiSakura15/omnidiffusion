#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Optional, Tuple
from itertools import islice

from datasets import load_dataset, Dataset, DownloadConfig


# 目前用到的 HF 数据集配置
DATASETS = {
    "coco": {
        "hf_id": "lmms-lab/COCO-Caption",
        "default_split": "val",
        "config_name": None,
    },
    "audiocaps": {
        "hf_id": "OpenSound/AudioCaps",
        "default_split": "train",
        "config_name": None,
    },
    "llava_cc3m": {
        "hf_id": "liuhaotian/LLaVA-CC3M-Pretrain-595K",
        "default_split": "train",
        "config_name": None,
    },
    # Salesforce/wikitext，使用 wikitext-103-raw-v1 配置
    "wikitext103": {
        "hf_id": "Salesforce/wikitext",
        "default_split": "train",
        "config_name": "wikitext-103-raw-v1",
    },
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_download_config(
    cache_dir: Optional[str],
    max_retries: int,
    num_proc: int,
    force_download: bool,
) -> DownloadConfig:
    """
    为当前机器的网络情况构造 DownloadConfig。

    - max_retries: 单个文件的最大重试次数
    - num_proc: 下载/parquet 处理时的并发进程数（网络差时建议 1）
    - force_download: 是否强制重新下载（默认 False，避免反复从 0 开始）
    """
    return DownloadConfig(
        cache_dir=cache_dir,
        max_retries=max_retries,
        num_proc=num_proc,
        force_download=force_download,
        resume_download=True,
        local_files_only=False,
        use_etag=True,
    )


def pick_split(
    hf_id: str,
    default_split: str,
    config_name: Optional[str],
    download_config: DownloadConfig,
) -> Tuple[str, Dataset]:
    """
    非 streaming 模式：优先加载 default_split（通常是 'train'），
    如果失败，尝试自动发现可用 split。
    """
    print(f"[INFO] 加载数据集: {hf_id}, config={config_name}")
    # 优先尝试直接加载指定 split
    try:
        if config_name is not None:
            ds = load_dataset(
                hf_id,
                config_name,
                split=default_split,
                download_config=download_config,
            )
        else:
            ds = load_dataset(
                hf_id,
                split=default_split,
                download_config=download_config,
            )
        print(f"[INFO] 使用 split='{default_split}'，样本数: {len(ds)}")
        return default_split, ds
    except Exception as e:
        print(f"[WARN] 直接加载 split='{default_split}' 失败: {e}")
        print("[INFO] 尝试自动探测可用 splits...")

        # 加载整个字典，再挑一个 split
        if config_name is not None:
            dsdict = load_dataset(
                hf_id,
                config_name,
                download_config=download_config,
            )
        else:
            dsdict = load_dataset(
                hf_id,
                download_config=download_config,
            )

        if default_split in dsdict:
            ds = dsdict[default_split]
            print(f"[INFO] 使用 split='{default_split}'，样本数: {len(ds)}")
            return default_split, ds

        # 否则取第一个 split 当 fallback
        split_name = list(dsdict.keys())[0]
        ds = dsdict[split_name]
        print(f"[INFO] 使用 split='{split_name}' (fallback)，样本数: {len(ds)}")
        return split_name, ds


def make_subset(
    ds: Dataset,
    subset_fraction: float,
    max_samples: Optional[int],
) -> Dataset:
    """
    非 streaming 模式下，根据 subset_fraction/max_samples 截取前 N 条。
    """
    n = len(ds)
    keep_n = n

    if subset_fraction < 1.0:
        keep_n = max(1, int(n * subset_fraction))

    if max_samples is not None:
        keep_n = min(keep_n, max_samples)

    keep_n = min(keep_n, n)

    print(f"[INFO] 原始样本数: {n}, 计划保留: {keep_n}")
    subset = ds.select(range(keep_n))
    return subset


def process_one_dataset_normal(
    name: str,
    cfg: dict,
    data_root: str,
    subset_fraction: float,
    max_samples: Optional[int],
    download_config: DownloadConfig,
):
    """
    普通（非 streaming）模式：
    - 会把该 split 的所有文件下载到本地 HF cache
    - 然后根据 subset_fraction/max_samples 截取并保存
    """
    hf_id = cfg["hf_id"]
    default_split = cfg.get("default_split", "train")
    config_name = cfg.get("config_name", None)

    split_name, ds = pick_split(hf_id, default_split, config_name, download_config)

    is_full = (subset_fraction >= 0.9999) and (max_samples is None)

    if is_full:
        save_dir = os.path.join(data_root, name, "full")
        print(f"[INFO] 保存全量数据集到: {save_dir}")
        ensure_dir(save_dir)
        ds.save_to_disk(save_dir)
        print(f"[DONE] {name} 全量数据保存完成。")
    else:
        print(
            f"[INFO] 生成子集: subset_fraction={subset_fraction}, "
            f"max_samples={max_samples}"
        )
        subset = make_subset(ds, subset_fraction, max_samples)
        save_dir = os.path.join(data_root, name, "subset")
        print(f"[INFO] 保存子集到: {save_dir}")
        ensure_dir(save_dir)
        subset.save_to_disk(save_dir)
        print(f"[DONE] {name} 子集保存完成。")


def process_one_dataset_streaming(
    name: str,
    cfg: dict,
    data_root: str,
    max_samples: Optional[int],
    download_config: DownloadConfig,
):
    """
    streaming 模式：
    - 使用 load_dataset(..., streaming=True)
    - 不会下载整个 parquet/tar，而是按样本流式读取
    - 只根据 max_samples 截取前 N 条，然后构造本地 Dataset 保存
    - 始终保存到 data/<name>/subset
    """
    hf_id = cfg["hf_id"]
    default_split = cfg.get("default_split", "train")
    config_name = cfg.get("config_name", None)

    if max_samples is None:
        max_samples = 1000
        print(
            f"[WARN] --streaming 模式下未指定 --max-samples，"
            f"默认改为 max_samples={max_samples}"
        )

    print(f"[INFO][STREAMING] 加载数据集: {hf_id}, config={config_name}, split={default_split}")
    try:
        if config_name is not None:
            ds_iter = load_dataset(
                hf_id,
                config_name,
                split=default_split,
                streaming=True,
                download_config=download_config,
            )
        else:
            ds_iter = load_dataset(
                hf_id,
                split=default_split,
                streaming=True,
                download_config=download_config,
            )
    except Exception as e:
        print(f"[ERROR][STREAMING] 加载 {hf_id} 失败: {e}")
        print("[ERROR][STREAMING] 暂时跳过该数据集。")
        return

    print(f"[INFO][STREAMING] 从流式数据集中取前 {max_samples} 条样本...")
    samples = list(islice(ds_iter, max_samples))
    print(f"[INFO][STREAMING] 实际获得样本数: {len(samples)}")

    if len(samples) == 0:
        print(f"[WARN][STREAMING] {name} 未获得任何样本，跳过保存。")
        return

    subset_ds = Dataset.from_list(samples)
    save_dir = os.path.join(data_root, name, "subset")
    print(f"[INFO][STREAMING] 保存子集到: {save_dir}")
    ensure_dir(save_dir)
    subset_ds.save_to_disk(save_dir)
    print(f"[DONE][STREAMING] {name} streaming 子集保存完成。")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "下载并预处理 OmniFlow 用到的 HF 数据集：\n"
            "  - lmms-lab/COCO-Caption\n"
            "  - OpenSound/AudioCaps\n"
            "  - liuhaotian/LLaVA-CC3M-Pretrain-595K\n"
            "  - Salesforce/wikitext (wikitext-103-raw-v1)\n"
            "支持 streaming 模式只拉少量样本做训练测试。"
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录（默认: 脚本所在目录的 data/）",
    )

    # 选择数据集
    parser.add_argument(
        "--coco",
        action="store_true",
        help="处理 lmms-lab/COCO-Caption",
    )
    parser.add_argument(
        "--audiocaps",
        action="store_true",
        help="处理 OpenSound/AudioCaps",
    )
    parser.add_argument(
        "--llava-cc3m",
        action="store_true",
        help="处理 liuhaotian/LLaVA-CC3M-Pretrain-595K",
    )
    parser.add_argument(
        "--wikitext",
        action="store_true",
        help="处理 Salesforce/wikitext (wikitext-103-raw-v1)",
    )

    # 子集相关（非 streaming 模式）
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=1.0,
        help=(
            "非 streaming 模式下：取多少比例的数据 (0,1]，默认 1.0 = 全量。"
            "例如 0.01 表示只取前 1%%。"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "每个数据集最多样本数（与 subset-fraction 同时生效，取两者最小值）。"
            "在 --streaming 模式下，只使用 max_samples，subset-fraction 被忽略。"
        ),
    )

    # 是否启用 streaming 模式
    parser.add_argument(
        "--streaming",
        action="store_true",
        help=(
            "启用 HuggingFace streaming 模式：不下载整个 split，只按样本流式读取。"
            "此时只根据 --max-samples 控制样本数，并保存到 subset/。"
        ),
    )

    # —— 下载相关配置（重点） ——————————————————————————
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help=(
            "自定义 HF 数据集缓存目录（默认使用环境变量 DATASETS_CACHE / HF_HOME 等）。"
        ),
    )
    parser.add_argument(
        "--download-max-retries",
        type=int,
        default=10,
        help="单个文件下载失败时的最大重试次数（默认 10）。",
    )
    parser.add_argument(
        "--download-num-proc",
        type=int,
        default=1,
        help="下载/parquet 处理的并发进程数，网络差时建议 1（默认 1）。",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "强制重新下载所有文件（默认关闭）。"
            "只有在缓存确实坏掉、且已经清空缓存目录的情况下再打开。"
        ),
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # data_root：默认 = 脚本所在目录的 data/
    if args.data_root is not None:
        data_root = args.data_root
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_root = os.path.join(script_dir, "data")

    os.makedirs(data_root, exist_ok=True)
    print(f"[INFO] 数据根目录: {data_root}")

    # 构造 DownloadConfig（统一传给所有 load_dataset 调用）
    download_config = build_download_config(
        cache_dir=args.hf_cache_dir,
        max_retries=args.download_max_retries,
        num_proc=args.download_num_proc,
        force_download=args.force_download,
    )
    print(
        f"[INFO] 下载配置: cache_dir={download_config.cache_dir}, "
        f"max_retries={download_config.max_retries}, "
        f"num_proc={download_config.num_proc}, "
        f"force_download={download_config.force_download}"
    )

    # 确定要处理哪些数据集
    selected = []
    if args.coco:
        selected.append("coco")
    if args.audiocaps:
        selected.append("audiocaps")
    if args.llava_cc3m:
        selected.append("llava_cc3m")
    if args.wikitext:
        selected.append("wikitext103")

    if not selected:
        # 用户没指定 → 默认全部
        selected = ["coco", "audiocaps", "llava_cc3m", "wikitext103"]
        print("[INFO] 未指定数据集，默认处理: coco, audiocaps, llava_cc3m, wikitext103")

    print(f"[INFO] 将处理的数据集: {selected}")
    print(
        f"[INFO] streaming={args.streaming}, "
        f"subset_fraction={args.subset_fraction}, "
        f"max_samples={args.max_samples}"
    )

    for name in selected:
        cfg = DATASETS[name]
        print("==========================================")
        print(f"[DATASET] 处理 {name} ({cfg['hf_id']})")
        print("==========================================")

        if args.streaming:
            process_one_dataset_streaming(
                name=name,
                cfg=cfg,
                data_root=data_root,
                max_samples=args.max_samples,
                download_config=download_config,
            )
        else:
            process_one_dataset_normal(
                name=name,
                cfg=cfg,
                data_root=data_root,
                subset_fraction=args.subset_fraction,
                max_samples=args.max_samples,
                download_config=download_config,
            )

    print("==========================================")
    print("[ALL DONE] 所选数据集处理完成。")
    print("目录结构示例：")
    print("  data/coco/full 或 data/coco/subset")
    print("  data/audiocaps/full 或 data/audiocaps/subset")
    print("  data/llava_cc3m/full 或 data/llava_cc3m/subset")
    print("  data/wikitext103/full 或 data/wikitext103/subset")
    print("==========================================")


if __name__ == "__main__":
    main()
