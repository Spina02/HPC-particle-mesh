#include "io.h"

// Recursively create all components of the given path (like `mkdir -p`).
// Returns 0 on success, -1 on error (with errno set accordingly).
static int mkdir_p(const char* path, mode_t mode) {
    if (path == NULL || *path == '\0') {
        errno = EINVAL;
        return -1;
    }

    // Work on a mutable copy
    char* tmp = strdup(path);
    if (!tmp) {
        return -1;
    }

    size_t len = strlen(tmp);
    if (len == 0) {
        free(tmp);
        errno = EINVAL;
        return -1;
    }

    // Remove trailing '/' to avoid creating empty components
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = '\0';
    }

    // Iterate over components and create them one by one
    for (char* p = tmp + 1; *p; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, mode) != 0 && errno != EEXIST) {
                free(tmp);
                return -1;
            }
            *p = '/';
        }
    }

    // Create the final directory
    if (mkdir(tmp, mode) != 0 && errno != EEXIST) {
        free(tmp);
        return -1;
    }

    free(tmp);
    return 0;
}

char* get_full_path(char* filename, char* type) {
    // Create the base output directory hierarchy if it doesn't exist
    if (mkdir_p(OUTPUT_DIR, 0755) != 0) {
        printf("Error while creating the base output directory %s\n", OUTPUT_DIR);
        return NULL;
    }

    size_t base_len = strlen(OUTPUT_DIR);
    char* output_dir = (char*) malloc(base_len + strlen(type) + 2);
    strcpy(output_dir, OUTPUT_DIR);
    strcat(output_dir, "/");
    strcat(output_dir, type);

    if (mkdir(output_dir, 0755) != 0 && errno != EEXIST) {
        printf("Error while creating the output directory %s\n", output_dir);
        free(output_dir);
        return NULL;
    }
    char* full_path = (char*) malloc(strlen(output_dir) + strlen(filename) + 2);
    strcpy(full_path, output_dir);
    strcat(full_path, "/");
    strcat(full_path, filename);
    free(output_dir);
    return full_path;
}

// ------------------------------------------------------------
//        functions to save the positions to a file
// ------------------------------------------------------------

// txt version
int save_positions_txt(Particles* particles, char* filename) {
    char* full_path = get_full_path(filename, "positions");
    if (full_path == NULL) return EXIT_FAILURE;
    
    FILE* fp = fopen(full_path, "w");
    if (fp == NULL) {
        printf("Error while opening the file %s\n", full_path);
        free(full_path);
        return EXIT_FAILURE;
    }
    for (uint i = 0; i < particles->N; i++) {
        fprintf(fp, REAL_FMT " " REAL_FMT "\n", particles->pos_col[i], particles->pos_row[i]);
    }
    
    fclose(fp);
    free(full_path);
    return EXIT_SUCCESS;
}

// bin version
int save_positions_bin(Particles* particles, char* filename) {
    char* full_path = get_full_path(filename, "positions");
    if (full_path == NULL) return EXIT_FAILURE;
    
    FILE* fp = fopen(full_path, "wb");
    if (fp == NULL) {
        printf("Error while opening the file %s\n", full_path);
        free(full_path);
        return EXIT_FAILURE;
    }

    int N = particles->N;
    
    for (int i = 0; i < N; i++) {
        fwrite(&particles->pos_col[i], sizeof(real_t), 1, fp);
        fwrite(&particles->pos_row[i], sizeof(real_t), 1, fp);
    }

    fclose(fp);
    free(full_path);
    return EXIT_SUCCESS;
}

// ------------------------------------------------------------
//        functions to save the status to a file
// ------------------------------------------------------------

// txt version
int save_status_txt(Mesh* mesh, char* filename) {
    char* full_path = get_full_path(filename, "status");
    if (full_path == NULL) return EXIT_FAILURE;
    
    FILE* fp = fopen(full_path, "w");
    if (fp == NULL) {
        printf("Error while opening the file %s\n", full_path);
        free(full_path);
        return EXIT_FAILURE;
    }
    for (uint i = 0; i < mesh->grid_size; i++) {
        fprintf(fp, REAL_FMT " " REAL_FMT " " REAL_FMT " " REAL_FMT "\n", mesh->density[i], mesh->pot[i], mesh->forces_x[i], mesh->forces_y[i]);
    }
    
    fclose(fp);
    free(full_path);
    return EXIT_SUCCESS;
}

// bin version
int save_status_bin(Mesh* mesh, char* filename) {
    char* full_path = get_full_path(filename, "status");
    if (full_path == NULL) return EXIT_FAILURE;
    
    FILE* fp = fopen(full_path, "wb");
    if (fp == NULL) {
        printf("Error while opening the file %s\n", full_path);
        free(full_path);
        return EXIT_FAILURE;
    }
    for (uint i = 0; i < mesh->grid_size; i++) {
        fwrite(&mesh->density[i], sizeof(real_t), 1, fp);
        fwrite(&mesh->pot[i], sizeof(real_t), 1, fp);
        fwrite(&mesh->forces_x[i], sizeof(real_t), 1, fp);
        fwrite(&mesh->forces_y[i], sizeof(real_t), 1, fp);
    }

    fclose(fp);
    free(full_path);
    return EXIT_SUCCESS;
}
