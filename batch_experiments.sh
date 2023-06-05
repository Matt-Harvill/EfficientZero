#!/bin/bash

# Define the arrays
env_type="NoFrameskip-v4"
envs=("Breakout" "MsPacman" "DemonAttack")

# Triple for loop
for env in "${envs[@]}"
do
    python main.py --env "$env$env_type" --case atari \
    --opr test --amp_type torch_amp --num_gpus 1 --load_model \
    --model_path "Final_Results/Models_100k/$env.p"
done