#!/bin/bash

#SBATCH --job-name=szs_FP
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
python regret_analysis_console.py --num_strategies=200 --num_emp_strategies=110 --num_samples=2000 --game_type=zero_sum --meta_method=FP

