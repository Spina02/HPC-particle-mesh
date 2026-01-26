#!/bin/bash

# =======================================================
#            HPC Execution - Strong Scaling
# =======================================================
# Strong scaling: fixed problem size, increasing threads

export JOB_NAME="pm-strong-scaling"
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
mkdir -p results
export TIMING_CSV=results/strong_scaling.csv

ACCOUNT=uTS25_Tornator_0
PARTITION=dcgp_usr_prod
EXEC=./bin/particle-mesh-hpc
CONF=params.conf

# Remove old CSV to start fresh
rm -f $TIMING_CSV

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

echo "Executing HPC job - Strong Scaling..."

for THREADS in 1 2 4 8 16 32 64 112; do
    export OMP_NUM_THREADS=${THREADS}
    sbatch --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=${THREADS} \
        --job-name=${JOB_NAME} \
        --export=ALL,TIMING_CSV=${TIMING_CSV} \
        --partition=$PARTITION \
        --account=$ACCOUNT \
        ./scripts/sbatch.sh $EXEC $CONF
done

echo "HPC job submitted"