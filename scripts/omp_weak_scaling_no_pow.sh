#!/bin/bash

# =======================================================
#            HPC Execution - Weak Scaling
# =======================================================
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
    srun -N1 -n1 -c8 --mem=8GB -p $PARTITION -A $ACCOUNT --time=00:05:00 make clean-hpc hpc
fi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Executing HPC job - Weak Scaling..."
echo "Base configuration: Npoints=${BASE_NPOINTS}, Grid=${BASE_NGRID_X}x${BASE_NGRID_Y}"

for THREADS in 1 2 4 8 16 32 64 112; do
    # Calculate the scaled parameters for weak scaling
    # Npoints scales linearly with the number of threads
    NPOINTS=$((BASE_NPOINTS * THREADS))
    
    # The grid scales with sqrt(THREADS) to maintain the point density per cell
    # Rounded to the nearest integer
    NGRID_X=$(echo "scale=0; $BASE_NGRID_X * sqrt($THREADS)" | bc | awk '{printf "%.0f", $1}')
    NGRID_Y=$(echo "scale=0; $BASE_NGRID_Y * sqrt($THREADS)" | bc | awk '{printf "%.0f", $1}')
    
    echo "Threads: ${THREADS} -> Npoints: ${NPOINTS}, Grid: ${NGRID_X}x${NGRID_Y}"
    
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