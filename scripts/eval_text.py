#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用文本评估脚本

输入：
  - 一个 JSON 文件 (--predictions)，格式为：
      [
        {"id": 0, "reference": "...", "prediction": "..."},
        ...
      ]

输出：
  - 在命令行打印各个指标结果
  - 可选地保存结果为 JSON (--output_json)

支持指标：
  - bleu   : BLEU-4
  - rouge  : ROUGE-L (F1)
  - meteor : METEOR (需要环境支持 nltk / evaluate 内部依赖)

依赖：
  pip install evaluate
"""

import json
import argparse
from pathlib import Path

import evaluate


def load_predictions(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    refs = []
    preds = []
    for item in data:
        ref = item.get("reference", "")
        pred = item.get("prediction", "")
        refs.append(ref)
        preds.append(pred)
    return refs, preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate text predictions with common NLP metrics")

    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON (output of generate_text_predictions.py)")
    parser.add_argument("--metrics", type=str, nargs="+", default=["bleu", "rouge"],
                        help="List of metrics to compute: bleu rouge meteor")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save metric results JSON")

    args = parser.parse_args()

    refs, preds = load_predictions(args.predictions)
    print(f"Loaded {len(refs)} prediction-reference pairs from {args.predictions}\n")

    results = {}

    if "bleu" in args.metrics:
        bleu_metric = evaluate.load("bleu")
        # evaluate 期望：references 是 list[list[str]]
        bleu_res = bleu_metric.compute(
            predictions=preds,
            references=[[r] for r in refs],
        )
        results["bleu"] = bleu_res["bleu"]
        print(f"BLEU-4: {results['bleu']:.4f}")

    if "rouge" in args.metrics:
        rouge_metric = evaluate.load("rouge")
        rouge_res = rouge_metric.compute(
            predictions=preds,
            references=refs,
            use_stemmer=True,
        )
        # 可以打印多个 ROUGE，这里重点 ROUGE-L
        results["rouge1"] = rouge_res.get("rouge1", 0.0)
        results["rouge2"] = rouge_res.get("rouge2", 0.0)
        results["rougeL"] = rouge_res.get("rougeL", 0.0)
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")

    if "meteor" in args.metrics:
        try:
            meteor_metric = evaluate.load("meteor")
            meteor_res = meteor_metric.compute(
                predictions=preds,
                references=refs,
            )
            results["meteor"] = meteor_res["meteor"]
            print(f"METEOR: {results['meteor']:.4f}")
        except Exception as e:
            print("[WARN] Failed to compute METEOR:", e)

    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metric results to {out_path}")


if __name__ == "__main__":
    main()
