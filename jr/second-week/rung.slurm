#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --error=errg.log
#SBATCH --output=outg.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160

python main.py
