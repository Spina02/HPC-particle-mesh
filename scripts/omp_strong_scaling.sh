#!/bin/bash

# =======================================================
#            HPC Execution - Strong Scaling
# =======================================================
# Strong scaling: fixed problem size, increasing threads

export JOB_NAME="pm-strong-scaling"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export TIMING_CSV=artifacts/strong_scaling.csv

make clean
make hpc

EXEC=./bin/particle-mesh-hpc
CONF=params.conf

# Remove old CSV to start fresh
rm -f $TIMING_CSV

echo "Executing HPC job - Strong Scaling..."

for THREADS in 1 2 4 8 16 32 64 112; do
    export OMP_NUM_THREADS=${THREADS}
    sbatch --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=${THREADS} \
        --job-name=${JOB_NAME} \
        --export=ALL,TIMING_CSV=${TIMING_CSV} \
        ./scripts/sbatch $EXEC $CONF
done

echo "HPC job submitted"