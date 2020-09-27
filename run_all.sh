#!/usr/bin/bash

sbatch -c 8 --gpus=1 amazon.py
sbatch -c 8 --gpus=1 amazon_ctr.py
sbatch -c 8 --gpus=1 amazon_ctr_2.py
sbatch -c 8 --gpus=1 amazon_ctr_255.py
sbatch -c 8 --gpus=1 adult.py
sbatch -c 8 --gpus=1 adult_ctr.py
sbatch -c 8 --gpus=1 appetency.py
sbatch -c 8 --gpus=1 appetency_ctr.py
sbatch -c 8 --gpus=1 upselling.py
sbatch -c 8 --gpus=1 upselling_ctr.py
sbatch -c 8 --gpus=1 kick.py
sbatch -c 8 --gpus=1 kick_ctr.py
sbatch -c 8 --gpus=1 higgs.py
sbatch -c 8 --gpus=1 higgs_ctr.py