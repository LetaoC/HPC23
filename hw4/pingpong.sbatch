#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=2GB
#SBATCH --job-name=pingpong
#SBATCH --output=pingpong.out
module purge
module load intel/19.1.2
module load openmpi/intel/4.1.1
mpirun -np 2 ./pingpong 1 2