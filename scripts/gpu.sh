#!/bin/bash

# -------------------------------------------------------
#            HPC Execution - GPU
# -------------------------------------------------------
# GPU: Single run, no scaling. Same problem size as serial for comparison.

export JOB_NAME="pm-gpu"
mkdir -p results
export TIMING_CSV=results/gpu_timing.csv

ACCOUNT=uTS25_Tornator
PARTITION=boost_usr_prod

EXEC=./bin/particle-mesh-gpu
CONF=params.conf

# Propagate USE_FLOAT (default 0) down to Makefile
USE_FLOAT_VALUE=${USE_FLOAT:-0}
echo ">>> Using USE_FLOAT=${USE_FLOAT_VALUE} for GPU build"

# Remove old CSV to start fresh
rm -f $TIMING_CSV
module purge
module load nvhpc

echo "Compiling on compute node..."
srun -N1 -n1 -c8 --mem=8GB -p $PARTITION -A $ACCOUNT --time=00:05:00 make clean-gpu gpu USE_FLOAT=${USE_FLOAT_VALUE}

NPOINTS=1048576
NGRID_X=256
NGRID_Y=256

echo "Running: Npoints=$NPOINTS, Grid=${NGRID_X}x${NGRID_Y}"

sbatch --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --gres=gpu:1 \
    --job-name=${JOB_NAME} \
    --export=ALL,TIMING_CSV=${TIMING_CSV} \
    --partition=$PARTITION \
    --account=$ACCOUNT \
    ./scripts/sbatch.sh $EXEC $CONF $NPOINTS $NGRID_X $NGRID_Y