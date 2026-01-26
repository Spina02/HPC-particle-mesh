#include "io.h"

char* get_full_path(char* filename, char* type) {
    // Create the base output directory first if it doesn't exist
    if (mkdir(OUTPUT_DIR, 0755) != 0 && errno != EEXIST) {
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
        fprintf(fp, "%lf %lf\n", particles->pos_col[i], particles->pos_row[i]);
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
        fwrite(&particles->pos_col[i], sizeof(double), 1, fp);
        fwrite(&particles->pos_row[i], sizeof(double), 1, fp);
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
        fprintf(fp, "%lf %lf %lf %lf\n", mesh->density[i], mesh->pot[i], mesh->forces_x[i], mesh->forces_y[i]);
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
        fwrite(&mesh->density[i], sizeof(double), 1, fp);
        fwrite(&mesh->pot[i], sizeof(double), 1, fp);
        fwrite(&mesh->forces_x[i], sizeof(double), 1, fp);
        fwrite(&mesh->forces_y[i], sizeof(double), 1, fp);
    }

    fclose(fp);
    free(full_path);
    return EXIT_SUCCESS;
}
