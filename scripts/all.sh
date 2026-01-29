#!/bin/bash

# Clean all previous results
make clean clean-out

# ---------------------------------------------------------
# 1. Compile CPU (Serial, Vec, HPC) on DCGP partition
# ---------------------------------------------------------
echo ">>> Compiling CPU targets on DCGP partition..."
# Compile serial, vec, and hpc on DCGP partition
# Propagate USE_FLOAT (default 0) down to Makefile
USE_FLOAT_VALUE=${USE_FLOAT:-0}
echo ">>> Using USE_FLOAT=${USE_FLOAT_VALUE} for CPU builds"
srun -N1 -n1 -c8 --mem=10GB -p dcgp_usr_prod -A uTS25_Tornator_0 --time=00:10:00 \
    bash -c "module purge && module load gcc/12.2.0 fftw/3.3.10--openmpi--4.1.6--gcc--12.2.0-spack0.22 && make -j8 serial vec hpc USE_FLOAT=${USE_FLOAT_VALUE}"

if [ $? -ne 0 ]; then
    echo "CPU Compilation failed!"
    exit 1
fi

# ---------------------------------------------------------
# 2. Compile GPU on Boost partition
# ---------------------------------------------------------
echo ">>> Compiling GPU target on Boost partition..."
echo ">>> Using USE_FLOAT=${USE_FLOAT_VALUE} for GPU build"
# Compile gpu on Boost partition
srun -N1 -n1 -c8 --mem=10GB -p boost_usr_prod -A uTS25_Tornator --time=00:10:00 \
    bash -c "module purge && module load nvhpc && make gpu USE_FLOAT=${USE_FLOAT_VALUE}"

if [ $? -ne 0 ]; then
    echo "GPU Compilation failed!"
    exit 1
fi

# ---------------------------------------------------------
# 3. Submit jobs
# ---------------------------------------------------------
export COMPILE=FALSE

echo ">>> All compilations successful. Submitting jobs..."

# Submit serial, vec, omp_weak_scaling, omp_strong_scaling, and gpu_scaling jobs
./scripts/serial.sh &&
./scripts/vec.sh &&
./scripts/omp_weak_scaling.sh &&
./scripts/omp_strong_scaling.sh &&
./scripts/gpu_scaling.sh

echo ">>> All jobs submitted successfully!"