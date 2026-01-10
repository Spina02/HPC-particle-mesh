#ifndef IO_H
#define IO_H

#include "global.h"
#include "pm.h"
#include <sys/stat.h>
#include <errno.h>

#ifdef TXT
    #define FORMAT "txt"
    #define save_positions(...) save_positions_txt(__VA_ARGS__)
    #define save_status(...) save_status_txt(__VA_ARGS__)
#else
    #define FORMAT "bin"
    #define save_positions(...) save_positions_bin(__VA_ARGS__)
    #define save_status(...) save_status_bin(__VA_ARGS__)
#endif

#define OUTPUT_DIR "artifacts"
#define OUTPUT_POSITIONS_FILE "positions_%d.%s"
#define OUTPUT_STATUS_FILE "status_%d.%s"

char* get_full_path(char* filename, char* type);

int save_positions_txt(Particles* particles, char* filename);
int save_positions_bin(Particles* particles, char* filename);

int save_status_txt(Mesh* mesh, char* filename);
int save_status_bin(Mesh* mesh, char* filename);

#endif // IO_H