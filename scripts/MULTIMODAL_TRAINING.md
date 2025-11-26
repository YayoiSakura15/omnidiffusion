cd /home/venus/qf/jiangqf/OmniFlows

# 1. 先生成预测（如果之前没跑）
python scripts/generate_text.py \
  --model_path "./models/OmniFlow-v0.5" \
  --checkpoint "./checkpoints/text_decoder_head_test/mlp_head/final_step458.pt" \
  --use_text_decoder_head \
  --text_decoder_head_dim 64 \
  --data_config "./config/data_config.json" \
  --split "val" \
  --batch_size 8 \
  --output_json "./checkpoints/text_decoder_head_test/mlp_head/val_predictions.json"

# 2. 计算 BLEU + ROUGE
python scripts/eval_text.py \
  --predictions "./checkpoints/text_decoder_head_test/mlp_head/val_predictions.json" \
  --metrics bleu rouge meteor \
  --output_json "./checkpoints/text_decoder_head_test/mlp_head/val_metrics.json"
