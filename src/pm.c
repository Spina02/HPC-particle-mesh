#include "pm.h"

double* estimate_density(Mesh* mesh, GridParams* params, Particles* particles) {
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = params->Ngrid[_col_];
    uint N_rows = params->Ngrid[_row_];
    double cellSize_col = params->BoxSize[_col_] / N_cols;
    double cellSize_row = params->BoxSize[_row_] / N_rows;
    double mass = particles->mass;

    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    double* restrict density = __builtin_assume_aligned(mesh->density, ALIGNMENT);
    double* restrict pos_col = __builtin_assume_aligned(particles->pos_col, ALIGNMENT);
    double* restrict pos_row = __builtin_assume_aligned(particles->pos_row, ALIGNMENT);
    #else
    double* density = mesh->density;
    double* pos_col = particles->pos_col;
    double* pos_row = particles->pos_row;
    #endif
    
    // initialize density array
    uint grid_size = N_cols * N_rows;
    
    #if defined(OMP) || defined(USE_GPU)
        #if defined(USE_GPU)
        #pragma acc parallel loop present(density[0:grid_size])
        #elif defined(OMP)
        #pragma omp parallel for simd schedule(static)
        #endif
        for (uint i = 0; i < grid_size; i++) {
            density[i] = 0.0;
        }
    #else
        memset(density, 0, grid_size * sizeof(double));
    #endif

        // iterate over particles
        #if defined(USE_GPU)
            #pragma acc parallel loop present(density[0:grid_size], pos_col[0:N_particles], pos_row[0:N_particles])
        #elif defined(VEC) && defined(ALIGNED) && defined(OMP)
            #pragma omp parallel for simd schedule(static) aligned(density : ALIGNMENT)
        #elif defined(VEC) && defined(ALIGNED)
            #pragma omp simd aligned(density : ALIGNMENT)
        #elif defined(VEC) && defined(OMP)
            #pragma omp parallel for simd schedule(static)
        #elif defined(VEC)
            #pragma omp simd
        #elif defined(OMP)
            #pragma omp parallel for schedule(static)
        #endif
        for (uint i = 0; i < N_particles; i++) {

            // get particle position in grid units
            double grid_col = pos_col[i] / cellSize_col;
            double grid_row = pos_row[i] / cellSize_row;

            int ic = (int) (grid_col + 0.5);
            int ir = (int) (grid_row + 0.5);

            int is[3] = {ic-1, ic, ic+1};
            int js[3] = {ir-1, ir, ir+1};

            double wcol[3], wrow[3];
            wcol[0] = TSC_weight(fabs(grid_col - is[0]));
            wcol[1] = TSC_weight(fabs(grid_col - is[1]));
            wcol[2] = TSC_weight(fabs(grid_col - is[2]));
            wrow[0] = TSC_weight(fabs(grid_row - js[0]));
            wrow[1] = TSC_weight(fabs(grid_row - js[1]));
            wrow[2] = TSC_weight(fabs(grid_row - js[2]));

            PRAGMA_UNROLL
            for (int jj = 0; jj < 3; jj++) {
                int row_idx = fast_mod(js[jj], N_rows);
                PRAGMA_UNROLL
                for (int ii = 0; ii < 3; ii++) {
                    int col_idx = fast_mod(is[ii], N_cols);
                    int cell_idx = row_idx * N_cols + col_idx;

                    #if defined(USE_GPU)
                    #pragma acc atomic update
                    #elif defined(OMP)
                    #pragma omp atomic update relaxed
                    #endif
                    density[cell_idx] += mass * wcol[ii] * wrow[jj];
                }
            }
        }
    return density;
}

void compute_potential(Mesh* mesh, vec2_t Ngrid, vec2d_t BoxSize) {
    
    // extract grid parameters for efficiency
    uint N_cols = Ngrid[_col_] / 2 + 1;
    uint N_rows = Ngrid[_row_];

    #ifdef USE_GPU
    cufftDoubleComplex* kDensity = mesh->kDensity;
    cufftDoubleComplex* kPot = mesh->kPot;
    #else
    fftw_complex* kDensity = mesh->kDensity;
    fftw_complex* kPot = mesh->kPot;
    #endif

    double norm_col = 2 * M_PI / BoxSize[_col_];
    double norm_row = 2 * M_PI / BoxSize[_row_];

    double half_delta_col = 0.5 * BoxSize[_col_] / Ngrid[_col_];
    double half_delta_row = 0.5 * BoxSize[_row_] / Ngrid[_row_];
    double inv_half_delta_col_sq = 1 / (half_delta_col * half_delta_col);
    double inv_half_delta_row_sq = 1 / (half_delta_row * half_delta_row);

    double k_col, k_row, green_func, sin_col, sin_row, sin2_col, sin2_row, denom;

    int idx;
    double G_prime_4_PI = -(4 * M_PI * G_prime);
    double N_rows_half = N_rows / 2.0;

    #if defined(USE_GPU)
        #pragma acc parallel loop present(kDensity[0:N_rows * N_cols], kPot[0:N_rows * N_cols])
    #elif defined(VEC) && defined(OMP)
        #pragma omp parallel for simd schedule(static) private(k_col, k_row, sin_col, sin_row, sin2_col, sin2_row, denom)
    #elif defined(OMP)
        #pragma omp parallel for schedule(static) private(k_col, k_row, sin_col, sin_row, sin2_col, sin2_row, denom)
    #elif defined(VEC)
        #pragma omp simd
    #endif
    for (uint i=0; i<N_rows; i++) {
        
        bool is_positive_row = (i <= N_rows_half);
        k_row = is_positive_row * i * norm_row + !is_positive_row * ((double)(N_rows - i)) * norm_row;

        sin_row = sin(k_row * half_delta_row);
        sin2_row = sin_row * sin_row;
        
        #if defined(USE_GPU)
            #pragma acc loop
        #elif defined(VEC)
            #pragma omp simd
        #endif
        for (uint j=0; j<N_cols; j++) {
            
            idx = i * N_cols + j;
            
            k_col = j * norm_col;

            sin_col = sin(k_col * half_delta_col);
            sin2_col = sin_col * sin_col;

            denom = sin2_col*inv_half_delta_col_sq + sin2_row*inv_half_delta_row_sq;
            denom = MAX(denom, 1e-14);
            
            green_func = G_prime_4_PI / denom;

            #ifdef USE_GPU
                kPot[idx].x = green_func * kDensity[idx].x;
                kPot[idx].y = green_func * kDensity[idx].y;
            #else
                kPot[idx][0] = green_func * kDensity[idx][0];
                kPot[idx][1] = green_func * kDensity[idx][1];
            #endif
        }
    }
}

void compute_forces(Mesh* mesh, vec2d_t BoxSize, vec2_t Ngrid) {
    uint prev_col, next_col, prev_row, next_row, row_idx, idx, i, j;
#if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    double* restrict pot = __builtin_assume_aligned(mesh->pot, ALIGNMENT);
    double* restrict forces_x = __builtin_assume_aligned(mesh->forces_x, ALIGNMENT);
    double* restrict forces_y = __builtin_assume_aligned(mesh->forces_y, ALIGNMENT);
#else
    double* pot = mesh->pot;
    double* forces_x = mesh->forces_x;
    double* forces_y = mesh->forces_y;
#endif

    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];

    double den_col_inv = N_cols / (2 * BoxSize[_col_]);
    double den_row_inv = N_rows / (2 * BoxSize[_row_]);

    #if defined(USE_GPU)
        #pragma acc parallel loop present(pot[0:N_rows*N_cols], forces_x[0:N_rows*N_cols], forces_y[0:N_rows*N_cols])
    #elif defined(OMP)
        #pragma omp parallel for schedule(static) private(prev_row, next_row, prev_col, next_col, row_idx, idx, i, j)
    #endif
    for (i = 0; i < N_rows; i++) {
        row_idx = i * N_cols;

        prev_row = fast_mod(i - 1 + N_rows, N_rows);
        next_row = fast_mod(i + 1, N_rows);
        
        #if defined(VEC) && defined(ALIGNED)
            #pragma omp simd aligned(pot : ALIGNMENT)
        #elif defined(VEC)
            #pragma omp simd
        #endif
        for (j = 0; j < N_cols; j++) {
            idx = row_idx + j;

            prev_col = fast_mod(j - 1 + N_cols, N_cols);
            next_col = fast_mod(j + 1, N_cols);

            // central difference approximation of the gradient
            forces_x[idx] = (pot[row_idx + prev_col] - pot[row_idx + next_col]) * den_col_inv;
            forces_y[idx] = (pot[prev_row * N_cols + j] - pot[next_row * N_cols + j]) * den_row_inv;
        }
    }
}


void interpolate_forces(Mesh* mesh, Particles* particles, vec2d_t BoxSize, vec2_t Ngrid) {
    if (particles == NULL || mesh == NULL) return;
    
    // initialize the forces on the particles
    uint N_particles = particles->N;
    #if defined(USE_GPU) && defined(OMP)
    #if defined(USE_GPU)
        #pragma acc parallel loop present(particles->acc_col[0:N_particles], particles->acc_row[0:N_particles])
    #elif defined(OMP)
        #pragma omp parallel for schedule(static)
    #endif
        for (uint i = 0; i < N_particles; i++) {
            particles->acc_col[i] = 0.0;
            particles->acc_row[i] = 0.0;
        }
    #else
        memset(particles->acc_col, 0, N_particles * sizeof(double));
        memset(particles->acc_row, 0, N_particles * sizeof(double));
    #endif

    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
        double* restrict pos_col = __builtin_assume_aligned(particles->pos_col, ALIGNMENT);
        double* restrict pos_row = __builtin_assume_aligned(particles->pos_row, ALIGNMENT);
        double* restrict acc_col = __builtin_assume_aligned(particles->acc_col, ALIGNMENT);
        double* restrict acc_row = __builtin_assume_aligned(particles->acc_row, ALIGNMENT);
        double* restrict forces_x = __builtin_assume_aligned(mesh->forces_x, ALIGNMENT);
        double* restrict forces_y = __builtin_assume_aligned(mesh->forces_y, ALIGNMENT);
    #else
        double* pos_col = particles->pos_col;
        double* pos_row = particles->pos_row;
        double* acc_col = particles->acc_col;
        double* acc_row = particles->acc_row;
        double* forces_x = mesh->forces_x;
        double* forces_y = mesh->forces_y;
    #endif

    // extract grid parameters for efficiency
    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];
    double cellSize_col_inv = N_cols / BoxSize[_col_];
    double cellSize_row_inv = N_rows / BoxSize[_row_];
    // mass is the same for all particles
    double mass_inv = 1.0 / particles->mass;
    
    double max_acc_col = 0.0;
    double max_acc_row = 0.0;
    
    // iterate over particles
    #if defined(USE_GPU)
        uint grid_size = N_cols * N_rows;
        #pragma acc parallel loop present(pos_col[0:N_particles], pos_row[0:N_particles], \
            acc_col[0:N_particles], acc_row[0:N_particles], \
            forces_x[0:grid_size], forces_y[0:grid_size]) \
            reduction(max:max_acc_col, max_acc_row)
    #elif defined(VEC) && defined(ALIGNED) && defined(OMP)
        #pragma omp parallel for simd schedule(guided) reduction(max:max_acc_col, max_acc_row) aligned(pos_col, pos_row, acc_col, acc_row : ALIGNMENT)
    #elif defined(VEC) && defined(ALIGNED)
        #pragma omp simd reduction(max:max_acc_col, max_acc_row) aligned(pos_col, pos_row, acc_col, acc_row : ALIGNMENT)
    #elif defined(VEC) && defined(OMP)
        #pragma omp parallel for simd schedule(guided) reduction(max:max_acc_col, max_acc_row)
    #elif defined(VEC)
        #pragma omp simd reduction(max:max_acc_col, max_acc_row)
    #elif defined(OMP)
        #pragma omp parallel for schedule(guided) reduction(max:max_acc_col, max_acc_row)
    #endif
    for (uint i = 0; i < N_particles; i++) {
        double acc_col_i = 0.0;
        double acc_row_i = 0.0;

        // get particle position in grid units
        double grid_col = pos_col[i] * cellSize_col_inv;
        double grid_row = pos_row[i] * cellSize_row_inv;

        int ic = (int) (grid_col + 0.5);
        int ir = (int) (grid_row + 0.5);

        int is[3] = {ic-1, ic, ic+1};
        int js[3] = {ir-1, ir, ir+1};

        double wcol[3], wrow[3];
        wcol[0] = TSC_weight(fabs(grid_col - is[0]));
        wcol[1] = TSC_weight(fabs(grid_col - is[1]));
        wcol[2] = TSC_weight(fabs(grid_col - is[2]));
        wrow[0] = TSC_weight(fabs(grid_row - js[0]));
        wrow[1] = TSC_weight(fabs(grid_row - js[1]));
        wrow[2] = TSC_weight(fabs(grid_row - js[2]));

        PRAGMA_UNROLL
        for (int jj = 0; jj < 3; jj++) {
            int row_idx = fast_mod(js[jj], N_rows);
            PRAGMA_UNROLL
            for (int ii = 0; ii < 3; ii++) {
                int col_idx = fast_mod(is[ii], N_cols);
                int cell_idx = row_idx * N_cols + col_idx;
                acc_col_i += forces_x[cell_idx] * wcol[ii] * wrow[jj];
                acc_row_i += forces_y[cell_idx] * wcol[ii] * wrow[jj];
            }
        }

        acc_row[i] = acc_row_i * mass_inv;
        acc_col[i] = acc_col_i * mass_inv;
        
        // needed for the timestep selector
        max_acc_col = MAX(max_acc_col, fabs(acc_col_i));
        max_acc_row = MAX(max_acc_row, fabs(acc_row_i));
    }

    particles->max_acc_col = max_acc_col;
    particles->max_acc_row = max_acc_row;
}

void kick_particles(Particles* particles, double dt) {
    uint N_particles = particles->N;
    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    double* restrict vel_col = __builtin_assume_aligned(particles->vel_col, ALIGNMENT);
    double* restrict vel_row = __builtin_assume_aligned(particles->vel_row, ALIGNMENT);
    double* restrict acc_col = __builtin_assume_aligned(particles->acc_col, ALIGNMENT);
    double* restrict acc_row = __builtin_assume_aligned(particles->acc_row, ALIGNMENT);
    #else
    double* vel_col = particles->vel_col;
    double* vel_row = particles->vel_row;
    double* acc_col = particles->acc_col;
    double* acc_row = particles->acc_row;
    #endif

    // auto vectorize 32, 16 byte vecs (versioned)
    #if defined(USE_GPU)
        #pragma acc parallel loop \
        present(vel_col[0:N_particles], vel_row[0:N_particles], \
                acc_col[0:N_particles], acc_row[0:N_particles])
    #elif defined(OMP) && defined(VEC) && defined(ALIGNED)
        #pragma omp parallel for simd schedule(static) aligned(vel_col, vel_row, acc_col, acc_row : ALIGNMENT) 
    #elif defined(OMP) && defined(VEC)
        #pragma omp parallel for simd schedule(static)
    #elif defined(OMP)
        #pragma omp parallel for schedule(static)
    #elif defined(VEC) && defined(ALIGNED)
        #pragma omp simd aligned(vel_col, vel_row, acc_col, acc_row : ALIGNMENT)
    #elif defined(VEC)
        #pragma omp simd
    #endif
    for (uint i = 0; i < N_particles; i++) {
        vel_col[i] += acc_col[i] * dt;
        vel_row[i] += acc_row[i] * dt;
    }
}

void drift_particles(Particles* particles, double dt, vec2d_t BoxSize) {
    uint N_particles = particles->N;
    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    double* restrict pos_col = __builtin_assume_aligned(particles->pos_col, ALIGNMENT);
    double* restrict pos_row = __builtin_assume_aligned(particles->pos_row, ALIGNMENT);
    double* restrict vel_col = __builtin_assume_aligned(particles->vel_col, ALIGNMENT);
    double* restrict vel_row = __builtin_assume_aligned(particles->vel_row, ALIGNMENT);
    #else
    double* pos_col = particles->pos_col;
    double* pos_row = particles->pos_row;
    double* vel_col = particles->vel_col;
    double* vel_row = particles->vel_row;
    #endif

    double ncols = BoxSize[_col_];
    double nrows = BoxSize[_row_];
    double pos_col_i, pos_row_i;
    // bool out_low, out_high;

    #if defined(USE_GPU)
        #pragma acc parallel loop present(pos_col[0:N_particles], pos_row[0:N_particles], \
                vel_col[0:N_particles], vel_row[0:N_particles])
    #elif defined(OMP) && defined(VEC) && defined(ALIGNED)
        #pragma omp parallel for simd schedule(guided) aligned(pos_col, pos_row, vel_col, vel_row : ALIGNMENT)
    #elif defined(OMP) && defined(VEC)
        #pragma omp parallel for simd schedule(guided)
    #elif defined(OMP)
        #pragma omp parallel for schedule(guided)
    #elif defined(VEC) && defined(ALIGNED)
        #pragma omp simd aligned(pos_col, pos_row, vel_col, vel_row : ALIGNMENT)
    #elif defined(VEC)
        #pragma omp simd
    #endif
    for (uint i = 0; i < N_particles; i++) {
        pos_col_i = pos_col[i] + vel_col[i] * dt;
        pos_row_i = pos_row[i] + vel_row[i] * dt;

        pos_col_i = fast_fmod(pos_col_i, ncols);
        pos_row_i = fast_fmod(pos_row_i, nrows);

        pos_col[i] = pos_col_i;
        pos_row[i] = pos_row_i;
    }
}

// Comparison function for qsort
#ifndef OMP
int compare_sort_items(const void *a, const void *b) {
    uint ca = ((SortItem *)a)->cell_index;
    uint cb = ((SortItem *)b)->cell_index;
    if (ca < cb) return -1;
    if (ca > cb) return 1;
    return 0;
}
#endif

void reorder_particles(Particles* p, GridParams* grid, double* tmp_arrays) {
    // should not be used with gpu:
    #ifdef USE_GPU
    printf("reorder_particles is on CPU, should not be used with gpu\n");
    exit(1);
    #endif
    uint N = p->N;
    SortItem *items = malloc(N * sizeof(SortItem));
    // Radix sort needs a temporary buffer of the same size as the original
    SortItem *items_tmp = malloc(N * sizeof(SortItem));
    
    double inv_cs_col = grid->Ngrid[_col_] / grid->BoxSize[_col_];
    double inv_cs_row = grid->Ngrid[_row_] / grid->BoxSize[_row_];
    uint stride = grid->Ngrid[_col_];

    // Calculate particles' cell index
    #if defined(OMP)
        #pragma omp parallel for schedule(static)
    #endif
    for(uint i=0; i<N; i++) {
        uint cx = (uint)(p->pos_col[i] * inv_cs_col);
        uint cy = (uint)(p->pos_row[i] * inv_cs_row);
        items[i].particle_index = i;
        items[i].cell_index = cy * stride + cx; 
    }

    // Order particles 
    #if defined(OMP)
    parallel_radix_sort(items, items_tmp, N);
    #else
    qsort(items, N, sizeof(SortItem), compare_sort_items);
    #endif

    // Reorder particles' arrays
    double *tmp_pos_col = tmp_arrays;
    double *tmp_pos_row = tmp_arrays + N;
    double *tmp_vel_col = tmp_arrays + 2 * N;
    double *tmp_vel_row = tmp_arrays + 3 * N;
    double *tmp_acc_col = tmp_arrays + 4 * N;
    double *tmp_acc_row = tmp_arrays + 5 * N;

    #if defined(OMP)
        #pragma omp parallel for schedule(static)
    #endif
    for(uint i=0; i<N; i++) {
        uint old_idx = items[i].particle_index;
        tmp_pos_col[i] = p->pos_col[old_idx];
        tmp_pos_row[i] = p->pos_row[old_idx];
        tmp_vel_col[i] = p->vel_col[old_idx];
        tmp_vel_row[i] = p->vel_row[old_idx];
        tmp_acc_col[i] = p->acc_col[old_idx];
        tmp_acc_row[i] = p->acc_row[old_idx];
    }

    // Copy back
    #if defined(OMP)
        #pragma omp parallel for schedule(static)
    #endif
    for(uint i=0; i<N; i++) {
        p->pos_col[i] = tmp_pos_col[i];
        p->pos_row[i] = tmp_pos_row[i];
        p->vel_col[i] = tmp_vel_col[i];
        p->vel_row[i] = tmp_vel_row[i];
        p->acc_col[i] = tmp_acc_col[i];
        p->acc_row[i] = tmp_acc_row[i];
    }

    // Free temporary arrays
    free(items);
    free(items_tmp);
}
