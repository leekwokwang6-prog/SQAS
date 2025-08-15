

for (( seed = 0; seed < 5; seed+=1)); do
    python sampling.py --seed $seed --device_name 'grid_16q' --task 'TFIM_8'
done

for (( seed = 0; seed < 5; seed+=1)); do
    python vqe_task_paralell.py --seed $seed --device_name 'grid_16q' --task 'TFIM_8'
done
