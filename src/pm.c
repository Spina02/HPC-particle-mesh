#include "pm.h"

#if defined(USE_OMP)
#define ATOMIC_UPDATE _Pragma("omp atomic update relaxed")
#elif defined(USE_GPU)
#define ATOMIC_UPDATE _Pragma("acc atomic update")
#else
#define ATOMIC_UPDATE
#endif

real_t* estimate_density(Mesh* mesh, GridParams* params, Particles* particles) {
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = params->Ngrid[_col_];
    uint N_rows = params->Ngrid[_row_];
    real_t cellSize_col = params->BoxSize[_col_] / N_cols;
    real_t cellSize_row = params->BoxSize[_row_] / N_rows;
    real_t mass = particles->mass;

    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    real_t* restrict density = __builtin_assume_aligned(mesh->density, ALIGNMENT);
    real_t* restrict pos_col = __builtin_assume_aligned(particles->pos_col, ALIGNMENT);
    real_t* restrict pos_row = __builtin_assume_aligned(particles->pos_row, ALIGNMENT);
    #else
    real_t* density = mesh->density;
    real_t* pos_col = particles->pos_col;
    real_t* pos_row = particles->pos_row;
    #endif
    
    // initialize density array
    uint grid_size = N_cols * N_rows;
    
    #if defined(USE_OMP) || defined(USE_GPU)
        #if defined(USE_GPU)
        #pragma acc parallel loop present(density[0:grid_size])
        #elif defined(USE_OMP)
        #pragma omp parallel for simd schedule(static)
        #endif
        for (uint i = 0; i < grid_size; i++) {
            density[i] = 0.0;
        }
    #else
        memset(density, 0, grid_size * sizeof(real_t));
    #endif

        // iterate over particles
        #if defined(USE_GPU)
            #pragma acc parallel loop \
                    present(density[0:grid_size], \
                            pos_col[0:N_particles], \
                            pos_row[0:N_particles])
        #elif defined(VEC) && defined(ALIGNED) && defined(USE_OMP)
            #pragma omp parallel for simd schedule(static) \
                    aligned(density : ALIGNMENT)
        #elif defined(VEC) && defined(ALIGNED)
            #pragma omp simd aligned(density : ALIGNMENT)
        #elif defined(VEC) && defined(USE_OMP)
            #pragma omp parallel for simd schedule(static)
        #elif defined(VEC)
            #pragma omp simd
        #elif defined(USE_OMP)
            #pragma omp parallel for schedule(static)
        #endif
        for (uint i = 0; i < N_particles; i++) {

            // get particle position in grid units
            real_t grid_col = pos_col[i] / cellSize_col;
            real_t grid_row = pos_row[i] / cellSize_row;

            int ic = (int) (grid_col + 0.5);
            int ir = (int) (grid_row + 0.5);

            // Pre-calculate indices
            int ic_m = fast_mod(ic - 1, N_cols);
            int ic_c = fast_mod(ic,     N_cols);
            int ic_p = fast_mod(ic + 1, N_cols);

            int ir_m = fast_mod(ir - 1, N_rows);
            int ir_c = fast_mod(ir,     N_rows);
            int ir_p = fast_mod(ir + 1, N_rows);

            // Pre-calculate weights
            real_t wc_m = TSC_weight(FABS(grid_col - (ic - 1)));
            real_t wc_c = TSC_weight(FABS(grid_col - ic));
            real_t wc_p = TSC_weight(FABS(grid_col - (ic + 1)));

            real_t wr_m = TSC_weight(FABS(grid_row - (ir - 1)));
            real_t wr_c = TSC_weight(FABS(grid_row - ir));
            real_t wr_p = TSC_weight(FABS(grid_row - (ir + 1)));

            // 2. Unroll the updates manually to avoid nested loops and array lookups

            // ir_m
            ATOMIC_UPDATE
            density[ir_m * N_cols + ic_m] += mass * wc_m * wr_m;
            ATOMIC_UPDATE
            density[ir_m * N_cols + ic_c] += mass * wc_c * wr_m;
            ATOMIC_UPDATE
            density[ir_m * N_cols + ic_p] += mass * wc_p * wr_m;
            
            // ir_c
            ATOMIC_UPDATE
            density[ir_c * N_cols + ic_m] += mass * wc_m * wr_c;
            ATOMIC_UPDATE
            density[ir_c * N_cols + ic_c] += mass * wc_c * wr_c;
            ATOMIC_UPDATE
            density[ir_c * N_cols + ic_p] += mass * wc_p * wr_c;

            // ir_p
            ATOMIC_UPDATE
            density[ir_p * N_cols + ic_m] += mass * wc_m * wr_p;
            ATOMIC_UPDATE
            density[ir_p * N_cols + ic_c] += mass * wc_c * wr_p;
            ATOMIC_UPDATE
            density[ir_p * N_cols + ic_p] += mass * wc_p * wr_p;
        }
    return density;
}

void compute_potential(Mesh* mesh, vec2_t Ngrid, vec2d_t BoxSize) {
    
    // extract grid parameters for efficiency
    uint N_cols = Ngrid[_col_] / 2 + 1;
    uint N_rows = Ngrid[_row_];
    
    // Use precision-agnostic complex type
    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    complex_t* restrict kDensity = __builtin_assume_aligned(mesh->kDensity, ALIGNMENT);
    complex_t* restrict kPot     = __builtin_assume_aligned(mesh->kPot, ALIGNMENT);
    #else
    complex_t* kDensity = mesh->kDensity;
    complex_t* kPot     = mesh->kPot;
    #endif

    real_t norm_col = (real_t)(2.0 * M_PI_T / BoxSize[_col_]);
    real_t norm_row = (real_t)(2.0 * M_PI_T / BoxSize[_row_]);

    real_t half_delta_col = (real_t)(0.5 * BoxSize[_col_] / Ngrid[_col_]);
    real_t half_delta_row = (real_t)(0.5 * BoxSize[_row_] / Ngrid[_row_]);
    real_t inv_half_delta_col_sq = 1 / (half_delta_col * half_delta_col);
    real_t inv_half_delta_row_sq = (real_t)(1.0 / (half_delta_row * half_delta_row));

    real_t k_col, k_row, green_func, sin_col, sin_row, sin2_col, sin2_row, denom;

    int idx;
    real_t G_prime_4_PI = (real_t)(-(4.0 * M_PI_T * G_prime));
    real_t N_rows_half = (real_t)N_rows * (real_t)0.5;

    #if defined(USE_GPU)
        #pragma acc parallel loop collapse(2) present(kDensity[0:N_rows * N_cols], kPot[0:N_rows * N_cols])
    #elif defined(VEC) && defined(USE_OMP)
        #pragma omp parallel for simd schedule(static) private(k_col, k_row, sin_col, sin_row, sin2_col, sin2_row, denom)
    #elif defined(USE_OMP)
        #pragma omp parallel for schedule(static) private(k_col, k_row, sin_col, sin_row, sin2_col, sin2_row, denom)
    #elif defined(VEC)
        #pragma omp simd
    #endif
    for (uint i=0; i<N_rows; i++) {
        #if defined(USE_GPU)
        for (uint j=0; j<N_cols; j++) {
        #endif
        
        bool is_positive_row = (i <= N_rows_half);
        k_row = is_positive_row * i * norm_row + !is_positive_row * ((real_t)(N_rows - i)) * norm_row;

        sin_row = SIN(k_row * half_delta_row);
        sin2_row = sin_row * sin_row;
        
        #if !defined(USE_GPU)
        #if defined(VEC)
            #pragma omp simd
        #endif
        for (uint j=0; j<N_cols; j++) {
        #endif
            
            idx = i * N_cols + j;
            
            k_col = j * norm_col;

            sin_col = SIN(k_col * half_delta_col);
            sin2_col = sin_col * sin_col;

            denom = sin2_col*inv_half_delta_col_sq + sin2_row*inv_half_delta_row_sq;
            denom = MAX(denom, (real_t)1e-7);
            
            green_func = G_prime_4_PI / denom;

            // Apply Green's function in Fourier space (precision & backend agnostic)
            C_RE(kPot[idx]) = green_func * C_RE(kDensity[idx]);
            C_IM(kPot[idx]) = green_func * C_IM(kDensity[idx]);
        }
    }
}

void compute_forces(Mesh* mesh, vec2d_t BoxSize, vec2_t Ngrid) {
    uint prev_col, next_col, prev_row, next_row, row_idx, idx, i, j;
#if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    real_t* restrict pot = __builtin_assume_aligned(mesh->pot, ALIGNMENT);
    real_t* restrict forces_x = __builtin_assume_aligned(mesh->forces_x, ALIGNMENT);
    real_t* restrict forces_y = __builtin_assume_aligned(mesh->forces_y, ALIGNMENT);
#else
    real_t* pot = mesh->pot;
    real_t* forces_x = mesh->forces_x;
    real_t* forces_y = mesh->forces_y;
#endif

    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];

    real_t den_col_inv = N_cols / (2 * BoxSize[_col_]);
    real_t den_row_inv = N_rows / (2 * BoxSize[_row_]);

    #if defined(USE_GPU)
        #pragma acc parallel loop collapse(2) present(pot[0:N_rows*N_cols], forces_x[0:N_rows*N_cols], forces_y[0:N_rows*N_cols])
    #elif defined(USE_OMP)
        #pragma omp parallel for schedule(static) private(prev_row, next_row, prev_col, next_col, row_idx, idx, i, j)
    #endif
    for (i = 0; i < N_rows; i++) {
        #if defined(USE_GPU)
            for (j = 0; j < N_cols; j++) {
        #endif
            row_idx = i * N_cols;
            prev_row = fast_mod(i - 1 + N_rows, N_rows);
            next_row = fast_mod(i + 1, N_rows);
            
        #if !defined(USE_GPU)
            #if defined(VEC) && defined(ALIGNED)
                #pragma omp simd aligned(pot : ALIGNMENT)
            #elif defined(VEC)
                #pragma omp simd
            #endif
            for (j = 0; j < N_cols; j++) {
        #endif
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
    #if defined(USE_GPU) && defined(USE_OMP)
    #if defined(USE_GPU)
        #pragma acc parallel loop present(particles->acc_col[0:N_particles], particles->acc_row[0:N_particles])
    #elif defined(USE_OMP)
        #pragma omp parallel for schedule(static)
    #endif
        for (uint i = 0; i < N_particles; i++) {
            particles->acc_col[i] = 0.0;
            particles->acc_row[i] = 0.0;
        }
    #else
        memset(particles->acc_col, 0, N_particles * sizeof(real_t));
        memset(particles->acc_row, 0, N_particles * sizeof(real_t));
    #endif

    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
        real_t* restrict pos_col = __builtin_assume_aligned(particles->pos_col, ALIGNMENT);
        real_t* restrict pos_row = __builtin_assume_aligned(particles->pos_row, ALIGNMENT);
        real_t* restrict acc_col = __builtin_assume_aligned(particles->acc_col, ALIGNMENT);
        real_t* restrict acc_row = __builtin_assume_aligned(particles->acc_row, ALIGNMENT);
        real_t* restrict forces_x = __builtin_assume_aligned(mesh->forces_x, ALIGNMENT);
        real_t* restrict forces_y = __builtin_assume_aligned(mesh->forces_y, ALIGNMENT);
    #else
        real_t* pos_col = particles->pos_col;
        real_t* pos_row = particles->pos_row;
        real_t* acc_col = particles->acc_col;
        real_t* acc_row = particles->acc_row;
        real_t* forces_x = mesh->forces_x;
        real_t* forces_y = mesh->forces_y;
    #endif

    // extract grid parameters for efficiency
    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];
    real_t cellSize_col_inv = N_cols / BoxSize[_col_];
    real_t cellSize_row_inv = N_rows / BoxSize[_row_];
    // mass is the same for all particles
    real_t mass_inv = 1.0 / particles->mass;
    
    real_t max_acc_col = 0.0;
    real_t max_acc_row = 0.0;
    
    // iterate over particles
    #if defined(USE_GPU)
        uint grid_size = N_cols * N_rows;
        #pragma acc parallel loop gang vector_length(128) \
            present(pos_col[0:N_particles], pos_row[0:N_particles], \
            acc_col[0:N_particles], acc_row[0:N_particles], \
            forces_x[0:grid_size], forces_y[0:grid_size]) \
            reduction(max:max_acc_col, max_acc_row)
    #elif defined(VEC) && defined(ALIGNED) && defined(USE_OMP)
        #pragma omp parallel for simd schedule(static) reduction(max:max_acc_col, max_acc_row) aligned(pos_col, pos_row, acc_col, acc_row : ALIGNMENT)
    #elif defined(VEC) && defined(ALIGNED)
        #pragma omp simd reduction(max:max_acc_col, max_acc_row) aligned(pos_col, pos_row, acc_col, acc_row : ALIGNMENT)
    #elif defined(VEC) && defined(USE_OMP)
        #pragma omp parallel for simd schedule(static) reduction(max:max_acc_col, max_acc_row)
    #elif defined(VEC)
        #pragma omp simd reduction(max:max_acc_col, max_acc_row)
    #elif defined(USE_OMP)
        #pragma omp parallel for schedule(static) reduction(max:max_acc_col, max_acc_row)
    #endif
    for (uint i = 0; i < N_particles; i++) {
        real_t acc_col_i = 0.0;
        real_t acc_row_i = 0.0;

        // get particle position in grid units
        real_t grid_col = pos_col[i] * cellSize_col_inv;
        real_t grid_row = pos_row[i] * cellSize_row_inv;

        int ic = (int) (grid_col + 0.5);
        int ir = (int) (grid_row + 0.5);

        // Precompute wrapped neighbor indices
        int ic_m = fast_mod(ic - 1, N_cols);
        int ic_c = fast_mod(ic,     N_cols);
        int ic_p = fast_mod(ic + 1, N_cols);

        int ir_m = fast_mod(ir - 1, N_rows);
        int ir_c = fast_mod(ir,     N_rows);
        int ir_p = fast_mod(ir + 1, N_rows);

        // Precompute weights
        real_t wc_m = TSC_weight(FABS(grid_col - (ic - 1)));
        real_t wc_c = TSC_weight(FABS(grid_col - ic));
        real_t wc_p = TSC_weight(FABS(grid_col - (ic + 1)));

        real_t wr_m = TSC_weight(FABS(grid_row - (ir - 1)));
        real_t wr_c = TSC_weight(FABS(grid_row - ir));
        real_t wr_p = TSC_weight(FABS(grid_row - (ir + 1)));

        // Unrolled 3x3 stencil, accumulate forces_x / forces_y

        // ir_m row
        int row_m = ir_m * N_cols;
        real_t w_mm = wc_m * wr_m;
        real_t w_cm = wc_c * wr_m;
        real_t w_pm = wc_p * wr_m;

        int idx_mm = row_m + ic_m;
        int idx_cm = row_m + ic_c;
        int idx_pm = row_m + ic_p;

        acc_col_i += forces_x[idx_mm] * w_mm;
        acc_row_i += forces_y[idx_mm] * w_mm;

        acc_col_i += forces_x[idx_cm] * w_cm;
        acc_row_i += forces_y[idx_cm] * w_cm;

        acc_col_i += forces_x[idx_pm] * w_pm;
        acc_row_i += forces_y[idx_pm] * w_pm;

        // ir_c row
        int row_c = ir_c * N_cols;
        real_t w_mc = wc_m * wr_c;
        real_t w_cc = wc_c * wr_c;
        real_t w_pc = wc_p * wr_c;

        int idx_mc = row_c + ic_m;
        int idx_cc = row_c + ic_c;
        int idx_pc = row_c + ic_p;

        acc_col_i += forces_x[idx_mc] * w_mc;
        acc_row_i += forces_y[idx_mc] * w_mc;

        acc_col_i += forces_x[idx_cc] * w_cc;
        acc_row_i += forces_y[idx_cc] * w_cc;

        acc_col_i += forces_x[idx_pc] * w_pc;
        acc_row_i += forces_y[idx_pc] * w_pc;

        // ir_p row
        int row_p = ir_p * N_cols;
        real_t w_mp = wc_m * wr_p;
        real_t w_cp = wc_c * wr_p;
        real_t w_pp = wc_p * wr_p;

        int idx_mp = row_p + ic_m;
        int idx_cp = row_p + ic_c;
        int idx_pp = row_p + ic_p;

        acc_col_i += forces_x[idx_mp] * w_mp;
        acc_row_i += forces_y[idx_mp] * w_mp;

        acc_col_i += forces_x[idx_cp] * w_cp;
        acc_row_i += forces_y[idx_cp] * w_cp;

        acc_col_i += forces_x[idx_pp] * w_pp;
        acc_row_i += forces_y[idx_pp] * w_pp;

        acc_row[i] = acc_row_i * mass_inv;
        acc_col[i] = acc_col_i * mass_inv;
        
        // needed for the timestep selector
        max_acc_col = MAX(max_acc_col, FABS(acc_col_i));
        max_acc_row = MAX(max_acc_row, FABS(acc_row_i));
    }

    particles->max_acc_col = max_acc_col;
    particles->max_acc_row = max_acc_row;
}

void kick_particles(Particles* particles, real_t dt) {
    uint N_particles = particles->N;
    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    real_t* restrict vel_col = __builtin_assume_aligned(particles->vel_col, ALIGNMENT);
    real_t* restrict vel_row = __builtin_assume_aligned(particles->vel_row, ALIGNMENT);
    real_t* restrict acc_col = __builtin_assume_aligned(particles->acc_col, ALIGNMENT);
    real_t* restrict acc_row = __builtin_assume_aligned(particles->acc_row, ALIGNMENT);
    #else
    real_t* vel_col = particles->vel_col;
    real_t* vel_row = particles->vel_row;
    real_t* acc_col = particles->acc_col;
    real_t* acc_row = particles->acc_row;
    #endif

    // auto vectorize 32, 16 byte vecs (versioned)
    #if defined(USE_GPU)
        #pragma acc parallel loop independent \
                present(vel_col[0:N_particles], vel_row[0:N_particles], \
                        acc_col[0:N_particles], acc_row[0:N_particles])
    #elif defined(USE_OMP) && defined(VEC) && defined(ALIGNED)
        #pragma omp parallel for simd schedule(static) aligned(vel_col, vel_row, acc_col, acc_row : ALIGNMENT) 
    #elif defined(USE_OMP) && defined(VEC)
        #pragma omp parallel for simd schedule(static)
    #elif defined(USE_OMP)
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

void drift_particles(Particles* particles, real_t dt, vec2d_t BoxSize) {
    uint N_particles = particles->N;
    #if defined(VEC) && defined(ALIGNED) && !defined(USE_GPU)
    real_t* restrict pos_col = __builtin_assume_aligned(particles->pos_col, ALIGNMENT);
    real_t* restrict pos_row = __builtin_assume_aligned(particles->pos_row, ALIGNMENT);
    real_t* restrict vel_col = __builtin_assume_aligned(particles->vel_col, ALIGNMENT);
    real_t* restrict vel_row = __builtin_assume_aligned(particles->vel_row, ALIGNMENT);
    #else
    real_t* pos_col = particles->pos_col;
    real_t* pos_row = particles->pos_row;
    real_t* vel_col = particles->vel_col;
    real_t* vel_row = particles->vel_row;
    #endif

    real_t ncols = BoxSize[_col_];
    real_t nrows = BoxSize[_row_];
    real_t pos_col_i, pos_row_i;
    // bool out_low, out_high;

    #if defined(USE_GPU)
        #pragma acc parallel loop independent \
                present(pos_col[0:N_particles], pos_row[0:N_particles], \
                        vel_col[0:N_particles], vel_row[0:N_particles])
    #elif defined(USE_OMP) && defined(VEC) && defined(ALIGNED)
        #pragma omp parallel for simd schedule(static) aligned(pos_col, pos_row, vel_col, vel_row : ALIGNMENT)
    #elif defined(USE_OMP) && defined(VEC)
        #pragma omp parallel for simd schedule(static)
    #elif defined(USE_OMP)
        #pragma omp parallel for schedule(static)
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
#ifndef USE_OMP
int compare_sort_items(const void *a, const void *b) {
    uint ca = ((SortItem *)a)->cell_index;
    uint cb = ((SortItem *)b)->cell_index;
    if (ca < cb) return -1;
    if (ca > cb) return 1;
    return 0;
}
#endif

#ifndef USE_GPU
void reorder_particles(Particles* restrict p, GridParams* restrict grid, real_t* restrict tmp_arrays) {
    // should not be used with gpu:
    // #ifdef USE_GPU
    // printf("reorder_particles is on CPU, should not be used with gpu\n");
    // exit(1);
    // #endif
    uint N = p->N;
    SortItem *items = (SortItem*) malloc(N * sizeof(SortItem));
    // Radix sort needs a temporary buffer of the same size as the original
    SortItem *items_tmp = (SortItem*) malloc(N * sizeof(SortItem));
    
    real_t inv_cs_col = grid->Ngrid[_col_] / grid->BoxSize[_col_];
    real_t inv_cs_row = grid->Ngrid[_row_] / grid->BoxSize[_row_];
    uint stride = grid->Ngrid[_col_];

    // Calculate particles' cell index
    #if defined(USE_GPU)
        #pragma acc parallel loop present(items[0:N], p->pos_col[0:N], p->pos_row[0:N])
    #elif defined(USE_OMP)
        #pragma omp parallel for schedule(static)
    #endif
    for(uint i=0; i<N; i++) {
        uint cx = (uint)(p->pos_col[i] * inv_cs_col);
        uint cy = (uint)(p->pos_row[i] * inv_cs_row);
        items[i].particle_index = i;
        items[i].cell_index = cy * stride + cx; 
    }

    // Order particles (CPU only: OpenMP radix sort or qsort fallback)
    #if defined(USE_OMP)
        parallel_radix_sort(items, items_tmp, N);
    #else
        qsort(items, N, sizeof(SortItem), compare_sort_items);
    #endif

    // Reorder particles' arrays
    real_t *tmp_pos_col = tmp_arrays;
    real_t *tmp_pos_row = tmp_arrays + N;
    real_t *tmp_vel_col = tmp_arrays + 2 * N;
    real_t *tmp_vel_row = tmp_arrays + 3 * N;
    real_t *tmp_acc_col = tmp_arrays + 4 * N;
    real_t *tmp_acc_row = tmp_arrays + 5 * N;

    #if defined(USE_GPU)
        #pragma acc parallel loop present(p->pos_col[0:N], p->pos_row[0:N], p->vel_col[0:N], p->vel_row[0:N], p->acc_col[0:N], p->acc_row[0:N], tmp_pos_col[0:N], tmp_pos_row[0:N], tmp_vel_col[0:N], tmp_vel_row[0:N], tmp_acc_col[0:N], tmp_acc_row[0:N])
    #elif defined(USE_OMP)
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
    #if defined(USE_GPU)
        #pragma acc parallel loop present(p->pos_col[0:N], p->pos_row[0:N], p->vel_col[0:N], p->vel_row[0:N], p->acc_col[0:N], p->acc_row[0:N], tmp_pos_col[0:N], tmp_pos_row[0:N], tmp_vel_col[0:N], tmp_vel_row[0:N], tmp_acc_col[0:N], tmp_acc_row[0:N])
    #elif defined(USE_OMP)
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
#else // GPU Implementation
void reorder_particles(Particles* restrict p, GridParams* restrict grid, real_t* restrict tmp_arrays) {
    uint N = p->N;
    uint grid_size = grid->Ngrid[_col_] * grid->Ngrid[_row_];

    // Grid parameters
    real_t inv_cs_col = grid->Ngrid[_col_] / grid->BoxSize[_col_];
    real_t inv_cs_row = grid->Ngrid[_row_] / grid->BoxSize[_row_];
    uint stride = grid->Ngrid[_col_];

    // Local aliases for particle data
    real_t* pos_col = p->pos_col;
    real_t* pos_row = p->pos_row;
    real_t* vel_col = p->vel_col;
    real_t* vel_row = p->vel_row;
    real_t* acc_col = p->acc_col;
    real_t* acc_row = p->acc_row;

    // Temporary arrays management
    // We use tmp_arrays as the GPU scratchpad. 
    // Layout: [pos_col | pos_row | vel_col | vel_row | acc_col | acc_row]
    // Each segment has length N.
    
    // Counts and offsets for sorting
    uint* counts  = (uint*) malloc(grid_size * sizeof(uint));
    uint* offsets = (uint*) malloc(grid_size * sizeof(uint));

    // GPU Region
    // Mappiamo tmp_arrays come CREATE perché non ci interessa il contenuto host,
    // lo usiamo solo come buffer temporaneo device-side.
    #pragma acc data create(counts[0:grid_size], offsets[0:grid_size], tmp_arrays[0:6*N]) \
                     present(pos_col[0:N], pos_row[0:N], \
                             vel_col[0:N], vel_row[0:N], \
                             acc_col[0:N], acc_row[0:N])
    {
        // 1. Zero out histogram
        #pragma acc parallel loop present(counts)
        for(uint i=0; i<grid_size; i++) {
            counts[i] = 0;
        }

        // 2. Compute histogram
        #pragma acc parallel loop present(counts, pos_col, pos_row)
        for(uint i=0; i<N; i++) {
            uint cx = (uint)(pos_col[i] * inv_cs_col);
            uint cy = (uint)(pos_row[i] * inv_cs_row);
            // Safety check per evitare segfault se particelle escono dal box
            if(cx < grid->Ngrid[_col_] && cy < grid->Ngrid[_row_]) {
                uint cell_idx = cy * stride + cx;
                #pragma acc atomic update
                counts[cell_idx]++;
            }
        }

        // 3. Prefix Sum (Serial on GPU is fine for small grids like 256^2 or 512^2)
        #pragma acc serial loop present(counts, offsets)
        for(uint i=0; i<grid_size; i++) {
            if (i == 0) offsets[i] = 0;
            else offsets[i] = offsets[i-1] + counts[i-1];
        }
        
        // Prepare counts to act as current insertion index
        #pragma acc parallel loop present(counts, offsets)
        for(uint i=0; i<grid_size; i++) counts[i] = offsets[i];

        // 4. Scatter (Move data to tmp_arrays)
        #pragma acc parallel loop present(counts, pos_col, pos_row, vel_col, vel_row, acc_col, acc_row, tmp_arrays)
        for(uint i=0; i<N; i++) {
            uint cx = (uint)(pos_col[i] * inv_cs_col);
            uint cy = (uint)(pos_row[i] * inv_cs_row);
            
            if(cx < grid->Ngrid[_col_] && cy < grid->Ngrid[_row_]) {
                uint cell_idx = cy * stride + cx;
                
                uint dest_idx;
                #pragma acc atomic capture
                dest_idx = counts[cell_idx]++;

                if (dest_idx < N) {
                    // Manual offset calculation into tmp_arrays
                    tmp_arrays[dest_idx]       = pos_col[i];
                    tmp_arrays[N + dest_idx]   = pos_row[i];
                    tmp_arrays[2*N + dest_idx] = vel_col[i];
                    tmp_arrays[3*N + dest_idx] = vel_row[i];
                    tmp_arrays[4*N + dest_idx] = acc_col[i];
                    tmp_arrays[5*N + dest_idx] = acc_row[i];
                }
            }
        }

        // 5. Copy back
        #pragma acc parallel loop present(pos_col, pos_row, vel_col, vel_row, acc_col, acc_row, tmp_arrays)
        for(uint i=0; i<N; i++) {
            pos_col[i] = tmp_arrays[i];
            pos_row[i] = tmp_arrays[N + i];
            vel_col[i] = tmp_arrays[2*N + i];
            vel_row[i] = tmp_arrays[3*N + i];
            acc_col[i] = tmp_arrays[4*N + i];
            acc_row[i] = tmp_arrays[5*N + i];
        }
    }

    free(counts);
    free(offsets);
}
#endif