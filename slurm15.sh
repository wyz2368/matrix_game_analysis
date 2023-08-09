#!/bin/bash

#SBATCH --job-name=real_world
#SBATCH --mail-user=wangyzhsrg@aol.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=14g
#SBATCH --time=10-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard

cd ${SLURM_SUBMIT_DIR}
python psro_real_world.py --num_iterations=30 --closed_method=dev --game_type="Transitive game"

