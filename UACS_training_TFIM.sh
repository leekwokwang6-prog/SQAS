#!/usr/bin/env bash

SCRIPT_PATH="UACS.py"


for seed in {0..4}; do
        python "$SCRIPT_PATH" \
          --seed "$seed" \
          --loop 2 \
          --hidden 128 \
          --bs 30 \
          --dropout 0.3 \
          --numCir_unlabel 30000 \
          --loss_weight 5 \
          --pre_epochs 200 \
          --infer_N 20 \
          --warmup_epochs 300\
          --top_k  500

done
