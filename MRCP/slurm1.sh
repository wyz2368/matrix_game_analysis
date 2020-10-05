#!/bin/bash

#SBATCH --job-name=matrix_ana
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=7g
#SBATCH --time=02-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python upper_bounded_method.py --num_strategies=100 --num_emp_strategies=40 --num_iter=10 --game_type=symmetric_zero_sum

