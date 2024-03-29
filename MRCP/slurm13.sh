#!/bin/bash

#SBATCH --job-name=real_regret
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=14g
#SBATCH --time=5-00:00:00
#SBATCH --account=wellman98
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python regret_analysis_console_real.py --num_emp_strategies=20 --num_samples=200 --game_type='Random game of skill' --meta_method=DO

