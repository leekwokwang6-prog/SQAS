#!/usr/bin/env bash


SCRIPT_PATH="Teacher-Student.py"

for seed in {0..4}; do
        python "$SCRIPT_PATH" \
          --seed "$seed" \
          --loop 2 \
          --hidden 128 \
          --bs 30 \
          --dropout 0.1 \
          --numCir_unlabel 40000 \
          --loss_weight 1 \
          --pre_epochs 200 \
          --warmup_epochs 300 \
          --ema_decay 0.99
done
