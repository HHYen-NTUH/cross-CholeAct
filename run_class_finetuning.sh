#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/path/to/cross-CholeAct/CholeCANE_Action/or/Cholec80_Action/csvs}"
MODEL_PATH="${MODEL_PATH:-/path/to/videomae_checkpoint.pth}"

# Fine-tune VideoMAE on cross-CholeAct.
python run_class_finetuning.py \
    --data_path "${DATA_DIR}" \
    --finetune "${MODEL_PATH}" \
    --output_dir ./output_finetune_8cls \
    --nb_classes 8 \
    --batch_size 4 \
    --num_workers 32 \
    --epochs 30 \
    --device cuda:0 \
    --save_ckpt_freq 10 \
    --log_dir ./logs_finetune

# Evaluation.
# The feature dump flags below are optional and are intended only for
# domain_gap_analysis.py.
python run_class_finetuning.py \
    --data_path "${DATA_DIR}" \
    --finetune "${MODEL_PATH}" \
    --output_dir ./testing_result \
    --nb_classes 8 \
    --batch_size 64 \
    --num_workers 40 \
    --device cuda:0 \
    --eval \
    --extract_eval_features \
    --feature_dump_level video \
    --feature_dump_dir ./testing_result/feature_dumps \
    --feature_dataset_source private \
    --feature_split_name test \
    --feature_save_logits


# Domain-Gap Analysis
# Run the domain-gap analysis with one feature dump for cross-CholeAct 
# and one for regenerated Cholec80-Action:
python domain_gap_analysis.py \
    --private_features ./testing_result/feature_dumps/test_private_video_features.npz \
    --public_features ./testing_result/feature_dumps/test_public_video_features.npz \
    --output_dir ./domain_gap_outputs