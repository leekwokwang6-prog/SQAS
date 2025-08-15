#!/usr/bin/env bash


SCRIPT_PATH="qas_framework.py"


for seed in {0..4}; do
        python "$SCRIPT_PATH" \
          --seed "$seed" \
          --run_id "$seed" \
          --task 'TFIM_8' \
          --device 'grid_16q'\
          --save_datasets './datasets/experiment_data/' \
          --exp_dir 'saved_models_Teacher-Student_grid_16q_TFIM_8/' \
          --n_qubit 8 \
          --model_path "predictor_Teacher-Student_seed${seed}.pth"

done
