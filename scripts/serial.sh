#!/bin/bash

# =======================================================
#                 Serial Execution
# =======================================================

export JOB_NAME="particle-mesh-serial"

make clean
make release

EXEC=./bin/particle-mesh
CONF=params.conf

echo "Executing serial job..."

sbatch --nodes=1 \
       --ntasks-per-node=1 \
       --cpus-per-task=1 \
       --job-name=particle-mesh-serial \
       ./scripts/sbatch $EXEC $CONF

echo "Serial job submitted"