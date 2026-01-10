CC = gcc
LDFLAGS = -lfftw3 -lm

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = build
BIN_DIR = bin
OUT_DIR = artifacts

# Directory targets
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

CFLAGS_COMMON = -Wall -Wextra -Wpedantic -Werror -I$(INC_DIR)

TARGET = $(BIN_DIR)/particle-mesh
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
HEADERS = $(wildcard $(INC_DIR)/*.h)

# Default target
all: release

debug: CFLAGS = $(CFLAGS_COMMON) -g -O0 -DDEBUG $(CFLAGS_EXTRA)
debug: $(TARGET)

release: CFLAGS = $(CFLAGS_COMMON) -O3 -DNDEBUG $(CFLAGS_EXTRA)
release: $(TARGET)

# Compile target
$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compile object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean targets
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) 

clean-out:
	rm -rf $(OUT_DIR)

# Run target
run: $(TARGET) params.conf
	./$(TARGET) params.conf

plot:
	python plot.py