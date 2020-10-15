#!/bin/bash

#SBATCH --job-name=upper_bound
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=14g
#SBATCH --time=05-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
#python upper_bounded_method.py --num_strategies=200 --num_emp_strategies=40 --num_iter=10 --game_type=symmetric_zero_sum
python upper_bounded_method.py --num_strategies=200 --num_emp_strategies=5 --num_iter=30 --game_type=kuhn

