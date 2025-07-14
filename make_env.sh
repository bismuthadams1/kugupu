#!/bin/bash

##SBATCH -A DCCADD
#SBATCH --mail-type=ALL
#SBATCH --export=ALL
#SBATCH -c 2
#SBATCH --mem=4G   # Set total memory for the job
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1      

mamba env create -f ./forked_minimal.yml -y

mamba run -n forked_kugupu pip install -e .
