#!/bin/bash

for horizon in 5; do 
    for seed in 2; do
	for lr in 0.001; do 
	    path=./results/diabcombolock/deepq/h=${horizon}-lr=${lr}-seed=${seed}
	    python -m baselines.run \
		   --alg=deepq \
		   --env=diabcombolock \
		   --num_timesteps=1e6 \
		   --save_path=$path \
		   --log_path=$path \
		   --horizon ${horizon} \
		   --layer_norm \
		   --prioritized_replay \
		   --exploration_fraction 0.001 \
		   --seed $seed \
		   --print_freq 10 \
		   --lr ${lr}
	done
    done
done
