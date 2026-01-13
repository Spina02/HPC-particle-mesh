CC = gcc
LDFLAGS = -lfftw3 -lm

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = build
OBJ_DIR_HPC = build_hpc
BIN_DIR = bin
OUT_DIR = artifacts

# Directory targets
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR_HPC):
	mkdir -p $(OBJ_DIR_HPC)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

CFLAGS_COMMON = -Wall -Wextra -Wpedantic -Werror -I$(INC_DIR)

CFLAGS_HPC_COMMON = $(CFLAGS_COMMON) -march=native -mtune=native -ffast-math -flto
CFLAGS_VEC = -DVEC -ftree-vectorize -funroll-loops -fopt-info-vec-missed #-fopt-info-vec-optimized #-fopt-info-vec-all
CFLAGS_OMP = -fopenmp

TARGET = $(BIN_DIR)/particle-mesh
TARGET_HPC = $(BIN_DIR)/particle-mesh-hpc

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
OBJS_HPC = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR_HPC)/%.o)
HEADERS = $(wildcard $(INC_DIR)/*.h)

# Default target
all: release hpc

debug: CFLAGS = $(CFLAGS_COMMON) -g -O0 -DDEBUG $(CFLAGS_EXTRA)
debug: $(TARGET)

release: CFLAGS = $(CFLAGS_COMMON) -O3 -DNDEBUG $(CFLAGS_EXTRA)
release: $(TARGET)

vec: CFLAGS = $(CFLAGS_HPC_COMMON) -O3 -DNDEBUG $(CFLAGS_VEC) $(CFLAGS_EXTRA)
vec: $(TARGET_HPC)

hpc: CFLAGS = $(CFLAGS_HPC_COMMON) -O3 -DNDEBUG $(CFLAGS_VEC) $(CFLAGS_EXTRA)
hpc: $(TARGET_HPC)

# Compile target
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile target HPC
$(TARGET_HPC): $(OBJS_HPC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile object files for HPC
$(OBJ_DIR_HPC)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR_HPC)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean targets
clean:
	rm -rf $(OBJ_DIR) $(OBJ_DIR_HPC) $(BIN_DIR) 

clean-out:
	rm -rf $(OUT_DIR)

# Run target
run: $(TARGET) params.conf
	./$(TARGET) params.conf

run-hpc: $(TARGET_HPC) params.conf
	./$(TARGET_HPC) params.conf

plot:
	python plot.py