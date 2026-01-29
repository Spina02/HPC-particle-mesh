# ----------------------------------------------------------------------------
#                              BUILD OPTIONS
# ----------------------------------------------------------------------------

# Set USE_FLOAT=1 to build with single precision
USE_FLOAT ?= 0
GPU_ARCH ?= cc80

# ----------------------------------------------------------------------------
#                              COMPILER & LINKER
# ----------------------------------------------------------------------------

CC  = gcc

# Precision‑dependent FFTW libraries
ifeq ($(USE_FLOAT),1)
    # --- SINGLE PRECISION ---
    FFTW_LIB     = -lfftw3f
    FFTW_OMP_LIB = -lfftw3f_omp
else
    # --- DOUBLE PRECISION ---
    FFTW_LIB     = -lfftw3
    FFTW_OMP_LIB = -lfftw3_omp
endif

LDFLAGS = $(FFTW_LIB) -lm

# ----------------------------------------------------------------------------
#                              DIRECTORIES
# ----------------------------------------------------------------------------

SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
OUT_DIR = artifacts
RES_DIR = results

OBJ_DIR_SERIAL = build/serial
OBJ_DIR_VEC    = build/vec
OBJ_DIR_HPC    = build/hpc
OBJ_DIR_GPU    = build/gpu

# C++/CUDA source for GPU radix sort
GPU_CPP_SRCS = $(SRC_DIR)/gpu_rsort.cpp
GPU_CPP_OBJS = $(OBJ_DIR_GPU)/gpu_rsort.o

# ----------------------------------------------------------------------------
#                              SOURCE FILES
# ----------------------------------------------------------------------------

SRCS     = $(wildcard $(SRC_DIR)/*.c)
HEADERS  = $(wildcard $(INC_DIR)/*.h)

OBJS_SERIAL = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR_SERIAL)/%.o)
OBJS_VEC    = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR_VEC)/%.o)
OBJS_HPC    = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR_HPC)/%.o)
OBJS_GPU    = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR_GPU)/%.o)

# ----------------------------------------------------------------------------
#                              TARGETS
# ----------------------------------------------------------------------------

TARGET_SERIAL = $(BIN_DIR)/particle-mesh-serial
TARGET_VEC    = $(BIN_DIR)/particle-mesh-vec
TARGET_HPC    = $(BIN_DIR)/particle-mesh-hpc
TARGET_GPU    = $(BIN_DIR)/particle-mesh-gpu

# ----------------------------------------------------------------------------
#                              COMPILER FLAGS
# ----------------------------------------------------------------------------

# Common flags for all builds
CFLAGS_COMMON = -Wall -Wextra -Wpedantic -Werror -I$(INC_DIR) -DPOW2GRID

ifeq ($(USE_FLOAT),1)
    CFLAGS_COMMON += -DUSE_FLOAT
endif

# HPC optimizations base
CFLAGS_HPC_BASE = $(CFLAGS_COMMON) -O3 -DNDEBUG \
                  -march=native -mtune=native \
                  -ffast-math -flto -fno-math-errno -fno-trapping-math

# Vectorization flags (-fopenmp-simd enables #pragma omp simd without full OpenMP)
CFLAGS_VEC = -DVEC -fopenmp-simd -ftree-vectorize -funroll-loops -mprefer-vector-width=512
# Optional: -fopt-info-vec-optimized -fopt-info-vec-missed

# OpenMP flags
CFLAGS_OMP = -fopenmp -DUSE_OMP
LDFLAGS_OMP = $(FFTW_OMP_LIB) -flto

# GPU/OpenACC flags (NVIDIA HPC SDK)
CFLAGS_GPU   = -I$(INC_DIR) -DPOW2GRID -O3 -fast -acc -gpu=$(GPU_ARCH),lineinfo -Minfo=accel -DUSE_GPU
LDFLAGS_GPU  = -cudalib=cufft -lm

# ----------------------------------------------------------------------------
#                              BUILD RULES
# ----------------------------------------------------------------------------

.PHONY: all debug serial vec omp hpc gpu test clean clean-out clean-serial clean-vec clean-hpc clean-gpu run-serial run-vec run-hpc run-gpu plot plot-results \
        sync-up sync-down get-results

all: serial vec hpc gpu

# ---------------------------- Serial Builds ---------------------------------

debug: CFLAGS = $(CFLAGS_COMMON) -g -O0 -DDEBUG -DOUTPUT_DIR=\"artifacts/serial\" $(CFLAGS_EXTRA)
debug: $(TARGET_SERIAL)

serial: CFLAGS = $(CFLAGS_COMMON) -O3 -DNDEBUG -DOUTPUT_DIR=\"artifacts/serial\" $(CFLAGS_EXTRA)
serial: $(TARGET_SERIAL)

# ----------------------------- HPC Builds -----------------------------------

vec: CC = gcc
vec: CFLAGS_VEC_BUILD = $(CFLAGS_HPC_BASE) -DALIGNED $(CFLAGS_VEC) -DOUTPUT_DIR=\"artifacts/vec\" $(CFLAGS_EXTRA)
vec: LDFLAGS = $(FFTW_LIB) -lm -flto
vec: $(TARGET_VEC)

omp: CFLAGS_HPC = $(CFLAGS_HPC_BASE) -DALIGNED $(CFLAGS_OMP) -DOUTPUT_DIR=\"artifacts/hpc\" $(CFLAGS_EXTRA)
omp: LDFLAGS = $(FFTW_LIB) -lm $(LDFLAGS_OMP)
omp: $(TARGET_HPC)

hpc: CFLAGS_HPC = $(CFLAGS_HPC_BASE) -DALIGNED $(CFLAGS_VEC) $(CFLAGS_OMP) -DOUTPUT_DIR=\"artifacts/hpc\" $(CFLAGS_EXTRA)
hpc: LDFLAGS = $(FFTW_LIB) -lm $(LDFLAGS_OMP)
hpc: $(TARGET_HPC)

# ----------------------------- GPU Builds -----------------------------------

gpu: CC = nvc
gpu: CFLAGS_GPU_BUILD = $(CFLAGS_GPU) -DALIGNED -DOUTPUT_DIR=\"artifacts/gpu\" $(CFLAGS_EXTRA)
gpu: LDFLAGS = $(LDFLAGS_GPU)
gpu: $(TARGET_GPU)

test:
	chmod +x scripts/*.sh && ./scripts/all.sh

test-float:
	chmod +x scripts/*.sh && USE_FLOAT=1 ./scripts/all.sh

# ----------------------------------------------------------------------------
#                              LINKING
# ----------------------------------------------------------------------------

$(TARGET_SERIAL): $(OBJS_SERIAL) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_VEC): $(OBJS_VEC) | $(BIN_DIR)
	$(CC) $(CFLAGS_VEC_BUILD) -o $@ $^ $(LDFLAGS)

$(TARGET_HPC): $(OBJS_HPC) | $(BIN_DIR)
	$(CC) $(CFLAGS_HPC) -o $@ $^ $(LDFLAGS)

$(TARGET_GPU): $(OBJS_GPU) | $(BIN_DIR)
	$(CC) $(CFLAGS_GPU_BUILD) -o $@ $^ $(LDFLAGS)

# ----------------------------------------------------------------------------
#                              COMPILATION
# ----------------------------------------------------------------------------

$(OBJ_DIR_SERIAL)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR_SERIAL)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR_VEC)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR_VEC)
	$(CC) $(CFLAGS_VEC_BUILD) -c $< -o $@

$(OBJ_DIR_HPC)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR_HPC)
	$(CC) $(CFLAGS_HPC) -c $< -o $@

$(OBJ_DIR_GPU)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR_GPU)
	$(CC) $(CFLAGS_GPU_BUILD) -c $< -o $@

# ----------------------------------------------------------------------------
#                              DIRECTORIES
# ----------------------------------------------------------------------------

$(OBJ_DIR_SERIAL):
	mkdir -p $@

$(OBJ_DIR_VEC):
	mkdir -p $@

$(OBJ_DIR_HPC):
	mkdir -p $@

$(OBJ_DIR_GPU):
	mkdir -p $@

$(BIN_DIR):
	mkdir -p $@

$(OUT_DIR):
	mkdir -p $@

# ----------------------------------------------------------------------------
#                              UTILITIES
# ----------------------------------------------------------------------------

clean:
	rm -rf build $(BIN_DIR)
clean-serial:
	rm -rf build/serial $(BIN_DIR)/particle-mesh-serial
clean-vec:
	rm -rf build/vec $(BIN_DIR)/particle-mesh-vec
clean-hpc:
	rm -rf build/hpc $(BIN_DIR)/particle-mesh-hpc
clean-gpu:
	rm -rf build/gpu $(BIN_DIR)/particle-mesh-gpu

clean-out:
	rm -rf $(OUT_DIR) $(RES_DIR)

run-serial: $(TARGET_SERIAL) params.conf
	./$(TARGET_SERIAL) params.conf

run-vec: $(TARGET_VEC) params.conf
	./$(TARGET_VEC) params.conf

run-hpc: $(TARGET_HPC) params.conf
	./$(TARGET_HPC) params.conf

run-gpu: $(TARGET_GPU) params.conf
	./$(TARGET_GPU) params.conf

plot:
	python plots/plot.py

plot-results:
	python plots/plot_results.py

# ----------------------------------------------------------------------------
#                              RSYNC (Leonardo)
# ----------------------------------------------------------------------------

LEO_HOST = login.leonardo.cineca.it
LEO_USER = aspinel1
LEO_PATH = ~/adv-hpc-project/
RSYNC_FLAGS = -avzP --exclude-from='.rsyncignore'

# Upload code to Leonardo
sync-up:
	rsync $(RSYNC_FLAGS) ./ $(LEO_USER)@$(LEO_HOST):$(LEO_PATH)

# Download all from Leonardo
sync-down:
	rsync $(RSYNC_FLAGS) $(LEO_USER)@$(LEO_HOST):$(LEO_PATH) ./

# Download $(RES_DIR) (CSV and plots) from remote
get-results:
	mkdir -p $(RES_DIR)
	rsync -avzP $(LEO_USER)@$(LEO_HOST):$(LEO_PATH)/$(RES_DIR)/ $(RES_DIR)/
