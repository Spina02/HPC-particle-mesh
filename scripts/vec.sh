#!/bin/bash

# -------------------------------------------------------
#            Vectorized (vec) execution on Leonardo
# -------------------------------------------------------
# Single run, no scaling. Same problem size as serial for comparison.

export JOB_NAME="pm-vec"
mkdir -p results
export TIMING_CSV=results/vec_timing.csv

ACCOUNT=uTS25_Tornator_0
PARTITION=dcgp_usr_prod
EXEC=./bin/particle-mesh-vec
CONF=params.conf

# Propagate USE_FLOAT (default 0) down to Makefile
USE_FLOAT_VALUE=${USE_FLOAT:-0}
echo ">>> Using USE_FLOAT=${USE_FLOAT_VALUE} for vec build (if compiled)"

# Optional: override problem size (Npoints NgridX NgridY [n_iter])
EXTRA_ARGS="${@:-}"

module purge
module load gcc/12.2.0
module load fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0-spack0.22

if [ "$COMPILE" = TRUE ]; then
    echo "Compiling vec build on compute node..."
    srun -N1 -n1 -c1 --mem=4GB -p $PARTITION -A $ACCOUNT --time=00:05:00 make clean-vec vec USE_FLOAT=${USE_FLOAT_VALUE}
fi

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Submitting vec job..."
sbatch --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --job-name=${JOB_NAME} \
    --export=ALL,TIMING_CSV=${TIMING_CSV} \
    --partition=$PARTITION \
    --account=$ACCOUNT \
    ./scripts/sbatch.sh $EXEC $CONF $EXTRA_ARGS

echo "Vec job submitted. Output in artifacts/pm_<jobid>.out"
