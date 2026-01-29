#!/bin/bash

# -------------------------------------------------------
#            HPC Execution - Weak Scaling
# -------------------------------------------------------
# Weak scaling: keep the load per thread constant
# Scaling the problem proportionally to the number of threads

export JOB_NAME="pm-weak-scaling"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
mkdir -p results
export TIMING_CSV=results/weak_scaling.csv

ACCOUNT=uTS25_Tornator_0
PARTITION=dcgp_usr_prod

EXEC=./bin/particle-mesh-hpc
CONF=params.conf

# Propagate USE_FLOAT (default 0) down to Makefile
USE_FLOAT_VALUE=${USE_FLOAT:-0}
echo ">>> Using USE_FLOAT=${USE_FLOAT_VALUE} for HPC weak-scaling build (if compiled)"

# Remove old CSV to start fresh
rm -f $TIMING_CSV

# Base values from params.conf
BASE_NPOINTS=262144
BASE_NGRID_X=256
BASE_NGRID_Y=256
N_ITER=100

module purge
module load gcc/12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0-spack0.22

if [ "$COMPILE" = TRUE ]; then
    echo "Compiling on compute node..."
    srun -N1 -n1 -c8 --mem=8GB -p $PARTITION -A $ACCOUNT --time=00:05:00 make clean-hpc hpc USE_FLOAT=${USE_FLOAT_VALUE}
fi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Executing HPC job - Weak Scaling..."
echo "Base configuration: Npoints=${BASE_NPOINTS}, Grid=${BASE_NGRID_X}x${BASE_NGRID_Y}"

for THREADS in 1 2 4 8 16 32 64 112; do
    
    # Calculate the grid target (for 112 threads, we use 128x128 grid)
    if [ "$THREADS" -eq 112 ]; then
        GRID_TARGET=128
    else
        GRID_TARGET=$THREADS
    fi

    # Calculate the grid dimensions based on GRID_TARGET (Rectangular logic)
    LOG2_TARGET=$(awk -v t=$GRID_TARGET 'BEGIN {print int(log(t)/log(2))}')
    EXP_X=$(( (LOG2_TARGET + 1) / 2 ))
    EXP_Y=$(( LOG2_TARGET / 2 ))
    MULT_X=$(( 1 << EXP_X ))
    MULT_Y=$(( 1 << EXP_Y ))
    
    NGRID_X=$(( BASE_NGRID_X * MULT_X ))
    NGRID_Y=$(( BASE_NGRID_Y * MULT_Y ))
    
    # Scale the number of points based on the grid target for consistency
    # for 112 threads, we use scale the number of points to 128
    NPOINTS=$(( BASE_NPOINTS * GRID_TARGET ))

    echo "Threads: ${THREADS} (Grid Target: ${GRID_TARGET}) -> Grid: ${NGRID_X}x${NGRID_Y}"
    
    export OMP_NUM_THREADS=${THREADS}
    sbatch --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=${THREADS} \
        --job-name=${JOB_NAME} \
        --export=ALL,TIMING_CSV=${TIMING_CSV} \
        --partition=$PARTITION \
        --account=$ACCOUNT \
        ./scripts/sbatch.sh $EXEC $CONF $NPOINTS $NGRID_X $NGRID_Y $N_ITER
done

echo "HPC job submitted"