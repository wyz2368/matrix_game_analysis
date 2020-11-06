#!/bin/bash

#SBATCH --job-name=real_regret
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=14g
#SBATCH --time=10-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
python regret_analysis_console_real.py --num_emp_strategies=110 --num_samples=200 --game_type='connect_four' --meta_method=DO

