#!/bin/bash

#$ -M nblancha@nd.edu
#$ -m ae
#$ -r y
#$ -q gpu@qa-xp-001
#$ -l gpu_card=1
#$ -N evaluate_model_on_human_stimuli

module load cuda/8.0

source ~/anaconda2/bin/activate prednet-env

gpu=$(nvidia-smi -q -d PIDS | awk 'BEGIN { gpu = 0; n = 0; inc = 1; } /Attached/ { n += $4; } /Processes/ && NF < 3 { if (inc == 1) { gpu++; } } /Processes/ && NF == 3 { inc = 0; } END { if (gpu >= n) { print(-1) } if (gpu < n) { print gpu; } }')
export CUDA_VISIBLE_DEVICES=$gpu
#if [ $gpu -eq -1 ]
#then
#   echo $gpu
#else
#   echo run
#fi

python vlog_evaluate.py $1
