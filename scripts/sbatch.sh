#!/bin/bash

# =======================================================
#           Common Parameters Among All Tests
# =======================================================

#SBATCH -t 00:10:00
#SBATCH --mem=0
#SBATCH --output=artifacts/pm_%j.out
#SBATCH --error=artifacts/pm_%j.err
#SBATCH --exclusive

# =======================================================
#      Parameters Dynamically Set by outer scripts
# =======================================================

# --account
# --partition
# --nodes
# --cpus-per-task
# --ntasks-per-node
# --gres

# =======================================================
#                 Environment Setup
# =======================================================

# Clean environment
module purge

# Load necessary modules
# For GCC build:
module load gcc/12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0-spack0.22

# For Intel build with MKL:
# module load intel-oneapi-compilers/2023.2.1
# module load intel-oneapi-mkl/2023.2.0

# Configure OpenMP
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OMP_DISPLAY_ENV=true

# Timing CSV (passed from outer script via --export or default)
export TIMING_CSV=${TIMING_CSV:-results/timing.csv}

# =======================================================
#                 Execution
# =======================================================

EXEC=$1
CONF=$2
shift 2  # Remove EXEC and CONF from arguments
ARGS="$@"  # All remaining arguments

echo "----------------------------------------------------"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Running on ${SLURM_NNODES} nodes"
echo "Total MPI tasks: ${SLURM_NTASKS}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "CPUs per task (OMP_NUM_THREADS): ${SLURM_CPUS_PER_TASK}"
echo "Program arguments: $EXEC $CONF $ARGS"
echo "----------------------------------------------------"

srun $EXEC $CONF $ARGS