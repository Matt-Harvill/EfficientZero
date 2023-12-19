#!/bin/bash

# Define the arrays
env_type="NoFrameskip-v4"
envs=("Breakout" "MsPacman" "DemonAttack")
searches=("2" "4" "8")

# Triple for loop
for search in "${searches[@]}"
do
    for env in "${envs[@]}"
    do
        python main.py --env "$env$env_type" --case atari \
        --opr test --amp_type torch_amp --num_gpus 1 --load_model \
        --model_path "Final_Results/Models_100k/$env.p" \
        --searches "$search"
    done
done