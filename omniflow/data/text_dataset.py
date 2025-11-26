"""
Text Dataset for Training Text Decoder Head
适配WikiText103和COCO Captions等数据格式
支持HuggingFace datasets格式
"""

import json
import random
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

try:
    from datasets import load_from_disk, Dataset as HFDataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    HFDataset = None


class WikiTextDataset(Dataset):
    """
    WikiText数据集加载器

    Args:
        data_path: WikiText文件路径 (.txt格式)
        tokenizer: tokenizer
        max_length: 最大序列长度
        min_length: 最小句子长度（单词数）
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        min_length: int = 10,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length

        # 读取WikiText文件
        print(f"Loading WikiText from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # 按空行分割段落
        paragraphs = raw_text.split('\n\n')

        # 过滤和处理
        self.texts = []
        for para in paragraphs:
            # 清理
            para = para.strip()

            # 跳过空段落
            if not para:
                continue

            # 跳过标题行（以=开头和结尾）
            if para.startswith('=') and para.endswith('='):
                continue

            # 跳过太短的段落
            if len(para.split()) < min_length:
                continue

            self.texts.append(para)

        print(f"Loaded {len(self.texts)} text segments from WikiText")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text,
        }


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions数据集加载器（Parquet格式）

    Args:
        data_path: Parquet文件路径
        tokenizer: tokenizer
        max_length: 最大序列长度
        min_length: 最小句子长度
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        min_length: int = 5,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length

        # 读取Parquet文件
        print(f"Loading COCO Captions from {data_path}...")
        df = pd.read_parquet(data_path)

        # 提取caption字段
        self.texts = []
        if 'caption' in df.columns:
            captions = df['caption'].tolist()
        elif 'text' in df.columns:
            captions = df['text'].tolist()
        else:
            # 尝试找到包含文本的列
            text_cols = [col for col in df.columns if 'caption' in col.lower() or 'text' in col.lower()]
            if text_cols:
                captions = df[text_cols[0]].tolist()
            else:
                raise ValueError(f"Cannot find caption/text column in {data_path}")

        # 过滤
        for caption in captions:
            if isinstance(caption, str):
                caption = caption.strip()
                if caption and len(caption.split()) >= min_length:
                    self.texts.append(caption)

        print(f"Loaded {len(self.texts)} captions from COCO")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text,
        }


class HFDatasetWrapper(Dataset):
    """
    HuggingFace Dataset包装器,适配本地保存的datasets格式
    支持多模态数据: text-only, image+text, audio+text

    Args:
        data_path: HF dataset目录路径 (使用load_from_disk加载)
        tokenizer: tokenizer
        text_field: 文本字段名称
        modality: 数据模态 ('text', 'image+text', 'audio+text')
        max_length: 最大序列长度
        min_length: 最小句子长度
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        text_field: str = 'text',
        modality: str = 'text',  # 'text', 'image+text', 'audio+text'
        max_length: int = 256,
        min_length: int = 5,
    ):
        super().__init__()

        if not HF_DATASETS_AVAILABLE:
            raise ImportError("Please install datasets: pip install datasets")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.text_field = text_field
        self.modality = modality

        # 加载HF dataset
        print(f"Loading HuggingFace dataset from {data_path} (modality={modality})...")
        self.dataset = load_from_disk(data_path)

        # 过滤有效样本
        self.valid_indices = []
        for idx, item in enumerate(self.dataset):
            # 检查是否有文本
            text = self._extract_text(item)
            if not text or len(text.split()) < min_length:
                continue

            # 检查多模态字段
            if modality == 'image+text':
                if 'image' not in item or item['image'] is None:
                    continue
            elif modality == 'audio+text':
                if 'audio' not in item or item['audio'] is None:
                    continue

            self.valid_indices.append(idx)

        print(f"Loaded {len(self.valid_indices)} valid {modality} samples from HF dataset")
        if len(self.valid_indices) == 0:
            print(f"Warning: No valid samples found! Available fields: {list(self.dataset[0].keys())}")

    def _extract_text(self, item):
        """从item中提取文本"""
        text = None

        # 1. 优先使用指定字段 self.text_field
        if self.text_field in item and item[self.text_field]:
            field_value = item[self.text_field]

            # 1.1 LLaVA 格式: conversations = List[{"from": "...", "value": "..."}]
            if self.text_field == 'conversations' and isinstance(field_value, list):
                texts = []
                for conv in field_value:
                    if isinstance(conv, dict) and 'value' in conv:
                        v = conv['value']
                        if isinstance(v, str):
                            texts.append(v)
                if texts:
                    text = " ".join(texts)

            # 1.2 一般的 list[str]，比如 COCO 的 answer: List[str]
            elif isinstance(field_value, list):
                # 过滤出字符串
                str_list = [v for v in field_value if isinstance(v, str)]
                if str_list:
                    # 这里取第一个 caption，也可以改成 " ".join(str_list)
                    text = str_list[0]

            # 1.3 普通的单字符串
            elif isinstance(field_value, str):
                text = field_value

        # 2. 如果上面没取到，再尝试常见字段（优先 caption / answer，其次 text，最后 question）
        if not text:
            # caption（如大部分图文数据集）
            if 'caption' in item and item['caption']:
                if isinstance(item['caption'], str):
                    text = item['caption']
                elif isinstance(item['caption'], list) and item['caption']:
                    # 兼容 list[str] 的情况
                    str_list = [v for v in item['caption'] if isinstance(v, str)]
                    if str_list:
                        text = str_list[0]

            # answer（例如 lmms-lab/COCO-Caption 的 captions 列表）
            elif 'answer' in item and isinstance(item['answer'], list) and item['answer']:
                str_list = [v for v in item['answer'] if isinstance(v, str)]
                if str_list:
                    text = str_list[0]

            # 纯文本数据集（如 wikitext 的 text 字段）
            elif 'text' in item and item['text']:
                if isinstance(item['text'], str):
                    text = item['text']

            # question（比如 COCO 里那个固定 prompt，优先级放到最后）
            elif 'question' in item and item['question']:
                if isinstance(item['question'], str):
                    text = item['question']

        if text and isinstance(text, str):
            return text.strip()
        return None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.dataset[real_idx]

        # 提取文本
        text = self._extract_text(item)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text,
            'modality': self.modality,
        }

        # 添加图像或音频
        if self.modality == 'image+text':
            img = item.get('image', None)
            # 处理字符串路径(LLaVA) vs PIL Image(COCO)
            if isinstance(img, str):
                from PIL import Image
                try:
                    img = Image.open(img).convert('RGB')
                except Exception:
                    img = None
            result['image'] = img

        elif self.modality == 'audio+text':
            audio = item.get('audio', None)
            audio_path = None

            # HF Audio 特征通常是一个 dict，里面有 'path' 字段指向缓存的音频文件
            if isinstance(audio, dict):
                audio_path = audio.get("path", None)
            elif isinstance(audio, str):
                # 兼容万一数据里直接就是 path 字符串
                audio_path = audio

            result["audio"] = audio_path
            
        return result




class MixedTextDataset(Dataset):
    """
    混合多个数据集，支持不同的采样权重
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            datasets: 数据集列表
            weights: 采样权重（如果为None，则均匀采样）
        """
        super().__init__()

        self.datasets = datasets

        # 设置权重
        if weights is None:
            weights = [1.0] * len(datasets)

        # 归一化权重
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]

        # 计算总长度
        self.total_length = sum(len(ds) for ds in datasets)

        print(f"\nMixed Dataset Statistics:")
        for i, (ds, weight) in enumerate(zip(self.datasets, self.weights)):
            print(f"  Dataset {i}: {len(ds)} samples, weight={weight:.2f}")
        print(f"  Total: {self.total_length} samples\n")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        # 根据权重随机选择一个数据集
        dataset_idx = random.choices(
            range(len(self.datasets)),
            weights=self.weights,
            k=1
        )[0]

        # 从选中的数据集随机采样
        dataset = self.datasets[dataset_idx]
        sample_idx = random.randint(0, len(dataset) - 1)

        return dataset[sample_idx]


def multimodal_collate_fn(batch):
    """
    自定义collate函数，处理多模态batch
    不同模态的样本可能有不同的字段(image, audio等)
    """
    # 收集所有keys
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())

    collated = {}

    for key in all_keys:
        # 对于所有样本都有的key，使用默认collate
        if all(key in item for item in batch):
            values = [item[key] for item in batch]

            # 特殊处理不同类型
            if key == 'text' or key == 'modality':
                # 字符串列表直接保存
                collated[key] = values
            elif key == 'image' or key == 'audio':
                # 图像/音频保存为列表（不stack，因为可能是PIL Image或dict）
                collated[key] = values
            else:
                # 其他(input_ids, attention_mask等)使用torch.stack
                import torch
                collated[key] = torch.stack(values, dim=0)
        else:
            # 对于只有部分样本有的key，填充None
            values = [item.get(key, None) for item in batch]
            collated[key] = values

    return collated


def create_dataloader_from_config(
    config: Dict,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    从配置创建数据加载器

    Args:
        config: 数据集配置，格式:
            {
                'datasets': [
                    {
                        'type': 'wikitext',
                        'path': '/path/to/train.txt',
                        'weight': 0.7
                    },
                    {
                        'type': 'coco',
                        'path': '/path/to/train.parquet',
                        'weight': 0.3
                    }
                ],
                'min_length': 10
            }
        tokenizer: tokenizer
        batch_size: batch大小
        max_length: 最大序列长度
        shuffle: 是否打乱
        num_workers: worker数量
        pin_memory: 是否pin memory

    Returns:
        dataloader
    """
    datasets = []
    weights = []

    min_length = config.get('min_length', 10)

    for ds_config in config['datasets']:
        ds_type = ds_config['type'].lower()
        ds_path = ds_config['path']
        ds_weight = ds_config.get('weight', 1.0)

        # 创建对应类型的数据集
        if ds_type == 'wikitext':
            dataset = WikiTextDataset(
                data_path=ds_path,
                tokenizer=tokenizer,
                max_length=max_length,
                min_length=min_length,
            )
        elif ds_type == 'coco':
            dataset = COCOCaptionsDataset(
                data_path=ds_path,
                tokenizer=tokenizer,
                max_length=max_length,
                min_length=min_length,
            )
        elif ds_type == 'hf_dataset':
            # 使用HF dataset包装器
            text_field = ds_config.get('text_field', 'text')
            modality = ds_config.get('modality', 'text')  # 'text', 'image+text', 'audio+text'
            dataset = HFDatasetWrapper(
                data_path=ds_path,
                tokenizer=tokenizer,
                text_field=text_field,
                modality=modality,
                max_length=max_length,
                min_length=min_length,
            )
        else:
            raise ValueError(f"Unknown dataset type: {ds_type}")

        datasets.append(dataset)
        weights.append(ds_weight)

    # 如果只有一个数据集，直接使用
    if len(datasets) == 1:
        final_dataset = datasets[0]
    else:
        # 否则创建混合数据集
        final_dataset = MixedTextDataset(datasets, weights)

    # 创建dataloader
    dataloader = DataLoader(
        final_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,  # 使用自定义collate函数
        pin_memory=pin_memory,
        drop_last=True,
    )

    return dataloader


if __name__ == '__main__':
    # 测试代码
    from transformers import AutoTokenizer

    print("Testing WikiTextDataset...")

    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 测试WikiText
    wikitext_path = '/home/venus/qf/OmniFlows/test_multimodal/data/WikiText103/train.txt'
    if Path(wikitext_path).exists():
        dataset = WikiTextDataset(
            data_path=wikitext_path,
            tokenizer=tokenizer,
            max_length=128,
            min_length=10,
        )

        print(f"Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"\nSample:")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  text: {sample['text'][:200]}...")

    # 测试COCO
    print("\n" + "="*60)
    print("Testing COCOCaptionsDataset...")

    coco_path = '/home/venus/qf/OmniFlows/test_multimodal/data/COCO_Captions_TextOnly/pair/train-00000-of-00001.parquet'
    if Path(coco_path).exists():
        dataset = COCOCaptionsDataset(
            data_path=coco_path,
            tokenizer=tokenizer,
            max_length=128,
            min_length=5,
        )

        print(f"Dataset size: {len(dataset)}")

        sample = dataset[0]
        print(f"\nSample:")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  text: {sample['text']}")

    # 测试混合数据集
    print("\n" + "="*60)
    print("Testing create_dataloader_from_config...")

    config = {
        'datasets': [
            {
                'type': 'wikitext',
                'path': wikitext_path,
                'weight': 0.7
            },
            {
                'type': 'coco',
                'path': coco_path,
                'weight': 0.3
            }
        ],
        'min_length': 5
    }

    dataloader = create_dataloader_from_config(
        config=config,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=128,
        num_workers=0,
    )

    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  Sample texts:")
    for i, text in enumerate(batch['text'][:2]):
        print(f"    [{i}] {text[:100]}...")

    print("\n✓ All tests passed!")
