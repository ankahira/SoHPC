#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --error=err.log
#SBATCH --output=out.log
#SBATCH --nodes=2

python main.py
