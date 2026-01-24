CC = gcc
LDFLAGS = -lfftw3 -lfftw3_omp -lm

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR_SERIAL = build/serial
OBJ_DIR_HPC = build/hpc
BIN_DIR = bin
OUT_DIR = artifacts

# Directory targets
$(OBJ_DIR_SERIAL):
	mkdir -p $(OBJ_DIR_SERIAL)

$(OBJ_DIR_HPC):
	mkdir -p $(OBJ_DIR_SERIAL) $(OBJ_DIR_HPC)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

CFLAGS_COMMON = -Wall -Wextra -Wpedantic -Werror -I$(INC_DIR) -DPOW2GRID

CFLAGS_HPC_COMMON = $(CFLAGS_COMMON) -march=native -mtune=native -ffast-math -flto -fno-math-errno -fno-trapping-math
CFLAGS_VEC = -DVEC -O3 -march=native -mtune=native -ftree-vectorize -funroll-loops -mprefer-vector-width=512
            # -fopt-info-vec-optimized -fopt-info-vec-missed

CFLAGS_OMP = -fopenmp -DOMP

TARGET = $(BIN_DIR)/particle-mesh
TARGET_HPC = $(BIN_DIR)/particle-mesh-hpc

SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR_SERIAL)/%.o)
OBJS_HPC = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR_HPC)/%.o)
HEADERS = $(wildcard $(INC_DIR)/*.h)

# Default target
all: release hpc

debug: CFLAGS = $(CFLAGS_COMMON) -g -O0 -DDEBUG $(CFLAGS_EXTRA)
debug: $(TARGET)

release: CFLAGS = $(CFLAGS_COMMON) -O3 -DNDEBUG $(CFLAGS_EXTRA)
release: $(TARGET)

vec: CFLAGS_HPC = $(CFLAGS_HPC_COMMON) -O3 -DNDEBUG $(CFLAGS_VEC) $(CFLAGS_EXTRA)
vec: LDFLAGS += -flto
vec: $(TARGET_HPC)

vec-aligned: CFLAGS_HPC = $(CFLAGS_HPC_COMMON) -O3 -DNDEBUG -DALIGNED $(CFLAGS_VEC) $(CFLAGS_EXTRA)
vec-aligned: LDFLAGS += -flto
vec-aligned: $(TARGET_HPC)

omp: CFLAGS_HPC = $(CFLAGS_HPC_COMMON) -O3 -DNDEBUG -DALIGNED $(CFLAGS_OMP) $(CFLAGS_EXTRA)
omp: LDFLAGS += -lfftw3_omp -flto
omp: $(TARGET_HPC)

hpc: CFLAGS_HPC = $(CFLAGS_HPC_COMMON) -O3 -DNDEBUG -DALIGNED $(CFLAGS_VEC) $(CFLAGS_OMP) $(CFLAGS_EXTRA)
hpc: LDFLAGS += -lfftw3_omp -flto
hpc: $(TARGET_HPC)

hpc-aligned: CFLAGS_HPC = $(CFLAGS_HPC_COMMON) -O3 -DNDEBUG -DALIGNED $(CFLAGS_VEC) $(CFLAGS_OMP) $(CFLAGS_EXTRA)
hpc-aligned: LDFLAGS += -lfftw3_omp
hpc-aligned: $(TARGET_HPC)

# Compile target
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile target HPC
$(TARGET_HPC): $(OBJS_HPC) | $(BIN_DIR)
	$(CC) $(CFLAGS_HPC) -o $@ $^ $(LDFLAGS)

# Compile object files
$(OBJ_DIR_SERIAL)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR_SERIAL)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile object files for HPC
$(OBJ_DIR_HPC)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR_HPC)
	$(CC) $(CFLAGS_HPC) -c $< -o $@

# Clean targets
clean:
	rm -rf $(OBJ_DIR_SERIAL) $(OBJ_DIR_HPC) $(BIN_DIR) 

clean-out:
	rm -rf $(OUT_DIR)

# Run target
run: $(TARGET) params.conf
	./$(TARGET) params.conf

run-hpc: $(TARGET_HPC) params.conf
	./$(TARGET_HPC) params.conf

plot:
	python plot.py

######################## Rsync targets #########################

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

# Download only the artifacts/output folder
get-results:
	rsync -avzP $(LEO_USER)@$(LEO_HOST):$(LEO_PATH)/artifacts/*.csv ./results