#!/bin/bash
## A template to generate all slurm files

#SBATCH --job-name=real_world
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=14g
#SBATCH --time=10-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

