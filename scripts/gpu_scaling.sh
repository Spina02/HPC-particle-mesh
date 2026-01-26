#!/bin/bash

# =======================================================
#            HPC Execution - GPU Scaling
# =======================================================
# GPU scaling: Increasing the problem size on a single device

export JOB_NAME="pm-gpu-scaling"
mkdir -p results
export TIMING_CSV=results/gpu_scaling.csv

ACCOUNT=uTS25_Tornator
PARTITION=boost_usr_prod

EXEC=./bin/particle-mesh-gpu
CONF=params.conf

# Remove old CSV to start fresh
rm -f $TIMING_CSV
module purge
module load nvhpc

if [ "$COMPILE" = TRUE ]; then
    echo "Compiling on compute node..."
    srun -N1 -n1 -c8 --mem=8GB -p $PARTITION -A $ACCOUNT --time=00:05:00 make clean-gpu gpu
fi

CONFIGS=(
    "131072 128 128"      # Very small (GPU likely underutilized)
    "1048576 256 256"     # Base (similar to current params.conf)
    "8388608 512 512"     # Medium
    "33554432 1024 1024"  # Large (GPU should shine here)
    # "134217728 2048 2048" # Very large (watch out for GPU memory!)
)

for CFG in "${CONFIGS[@]}"; do

    SET=($CFG)
    NPOINTS=${SET[0]}
    NGRID_X=${SET[1]}
    NGRID_Y=${SET[2]}

    echo "Running: Npoints=$NPOINTS, Grid=${NGRID_X}x${NGRID_Y}, N_ITER=$N_ITER"

    sbatch --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=8 \
        --gres=gpu:1 \
        --job-name=${JOB_NAME} \
        --export=ALL,TIMING_CSV=${TIMING_CSV} \
        --partition=$PARTITION \
        --account=$ACCOUNT \
        ./scripts/sbatch.sh $EXEC $CONF $NPOINTS $NGRID_X $NGRID_Y
done