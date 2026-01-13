#include "pm.h"

//TODO: enhance this for vectorization
double* estimate_density(Mesh* mesh, GridParams* params, Particles* particles) {
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = params->Ngrid[_col_];
    uint N_rows = params->Ngrid[_row_];
    double cellSize_col = params->BoxSize[_col_] / N_cols;
    double cellSize_row = params->BoxSize[_row_] / N_rows;
    double mass = particles->mass;
    double* restrict density = mesh->density;

    // initialize density array
    memset(density, 0, N_cols * N_rows * sizeof(double));

    // iterate over particles
    for (uint i = 0; i < N_particles; i++) {

        // get particle position in grid units
        double grid_col = particles->pos_col[i] / cellSize_col;
        double grid_row = particles->pos_row[i] / cellSize_row;

        // TSC has support up to 1.5 cells, so we need to check a larger stencil
        // Find the range of cells that could be influenced by this particle
        int col_min = (int)floor(grid_col - 1.5);
        int col_max = (int)floor(grid_col + 1.5);
        int row_min = (int)floor(grid_row - 1.5);
        int row_max = (int)floor(grid_row + 1.5);

        for (int c = col_min; c <= col_max; c++) {
            // get the column neighbor center
            double col_neighbor = (double)c + 0.5;
            // get relative position and distance from the grid cell center
            double dist_col = fabs(grid_col - col_neighbor);
            // skip if outside TSC support (distance > 1.5)
            if (dist_col >= 1.5) continue;
            // get the column index
            int col_idx = fast_mod(col_neighbor, N_cols);

            for (int r = row_min; r <= row_max; r++) {
                // get the row neighbor center
                double row_neighbor = (double)r + 0.5;
                // get relative position and distance from the grid cell center
                double dist_row = fabs(grid_row - row_neighbor);
                // skip if outside TSC support (distance > 1.5)
                if (dist_row >= 1.5) continue;
                // get the row index
                int row_idx = fast_mod(row_neighbor, N_rows);

                // row-major layout: y-major rows, x-major columns
                int cell_idx = row_idx * N_cols + col_idx;
                // random access -> issue for vectorization
                density[cell_idx] += mass * TSC_weight(dist_col)*TSC_weight(dist_row);
            }
        }
    }

    mesh->density = density;
    return density;
}

//TODO: enhance this for vectorization
void compute_potential(Mesh* mesh, vec2_t Ngrid, vec2d_t BoxSize) {
    
    // extract grid parameters for efficiency
    uint N_cols = Ngrid[_col_] / 2 + 1;
    uint N_rows = Ngrid[_row_];

    double norm_col = 2 * M_PI / BoxSize[_col_];
    double norm_row = 2 * M_PI / BoxSize[_row_];

    double half_delta_col = 0.5 * BoxSize[_col_] / Ngrid[_col_];
    double half_delta_row = 0.5 * BoxSize[_row_] / Ngrid[_row_];
    double half_delta_col_sq = half_delta_col * half_delta_col;
    double half_delta_row_sq = half_delta_row * half_delta_row;

    double k_col, k_row, green_func, sin_col, sin_row, sin2_col, sin2_row, denom;

    int idx;
    double G_prime_4_PI = -(4 * M_PI * G_prime);

    for (uint i=0; i<N_rows; i++) {
        
        if (i <= N_rows/2)
            k_row = i * norm_row;
        else
            k_row = -((double) (N_rows - i)) * norm_row;

        sin_row = sin(k_row * half_delta_row);
        sin2_row = sin_row * sin_row;
        
        for (uint j=0; j<N_cols; j++) {
            
            idx = i * N_cols + j;
            
            k_col = j * norm_col;

            sin_col = sin(k_col * half_delta_col);
            sin2_col = sin_col * sin_col;

            denom = sin2_col/half_delta_col_sq + sin2_row/half_delta_row_sq;
            denom = MAX(denom, 1e-14);
            
            green_func = G_prime_4_PI / denom;

            mesh->kPot[idx][0] = green_func * mesh->kDensity[idx][0];
            mesh->kPot[idx][1] = green_func * mesh->kDensity[idx][1];
        }
    }
}

//TODO: enhance this for vectorization
void compute_forces(Mesh* mesh, vec2d_t BoxSize, vec2_t Ngrid) {
    uint prev_col, next_col, prev_row, next_row, row_idx, idx, i, j;
    double* pot = mesh->pot;

    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];

    double den_col_inv = N_cols / (2 * BoxSize[_col_]);
    double den_row_inv = N_rows / (2 * BoxSize[_row_]);

    for (i = 0; i < N_rows; i++) {
        row_idx = i * N_cols;

        prev_row = (i - 1 + N_rows) % N_rows;
        next_row = (i + 1) % N_rows;
        
        for (j = 0; j < N_cols; j++) {
            idx = row_idx + j;

            prev_col = (j - 1 + N_cols) % N_cols;
            next_col = (j + 1) % N_cols;

            // central difference approximation of the gradient
            mesh->forces_x[idx] = (pot[row_idx + prev_col] - pot[row_idx + next_col]) * den_col_inv;
            mesh->forces_y[idx] = (pot[prev_row * N_cols + j] - pot[next_row * N_cols + j]) * den_row_inv;
        }
    }
}


void interpolate_forces(Mesh* mesh, Particles* particles, vec2d_t BoxSize, vec2_t Ngrid) {
    if (particles == NULL || mesh == NULL) return;
    
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];
    double cellSize_col_inv = N_cols / BoxSize[_col_];
    double cellSize_row_inv = N_rows / BoxSize[_row_];
    double* forces_x = mesh->forces_x;
    double* forces_y = mesh->forces_y;
    // mass is the same for all particles
    double mass_inv = 1.0 / particles->mass;
    
    // initialize the forces on the particles
    memset(particles->acc_col, 0, N_particles * sizeof(double));
    memset(particles->acc_row, 0, N_particles * sizeof(double));
    particles->max_acc_col = 0.0;
    particles->max_acc_row = 0.0;
    
    // iterate over particles
    for (uint i = 0; i < N_particles; i++) {
        double acc_col_i = 0.0;
        double acc_row_i = 0.0;

        // get particle position in grid units
        double grid_col = particles->pos_col[i] * cellSize_col_inv;
        double grid_row = particles->pos_row[i] * cellSize_row_inv;

        // TSC has support up to 1.5 cells, so we need to check a larger stencil
        // Find the range of cells that could be influenced by this particle
        int col_min = (int)floor(grid_col - 1.5);
        int col_max = (int)floor(grid_col + 1.5);
        int row_min = (int)floor(grid_row - 1.5);
        int row_max = (int)floor(grid_row + 1.5);

        for (int c = col_min; c <= col_max; c++) {
            // get the column neighbor center
            double col_neighbor = (double)c + 0.5;
            // get relative position and distance from the grid cell center
            double dist_col = fabs(grid_col - col_neighbor);
            if (dist_col >= 1.5) continue;
            // get the column index
            int col_idx = fast_fmod(col_neighbor, N_cols);

            for (int r = row_min; r <= row_max; r++) {

                double row_neighbor = (double)r + 0.5;
                // get relative position and distance from the grid cell center
                double dist_row = fabs(grid_row - row_neighbor);
                // skip if outside TSC support (distance > 1.5)
                if (dist_row >= 1.5) continue;
                // get the row index
                int row_idx = fast_fmod(row_neighbor, N_rows);
                
                // row-major layout: y-major rows, x-major columns
                int cell_idx = row_idx * N_cols + col_idx;
                
                acc_col_i += forces_x[cell_idx]*mass_inv * TSC_weight(dist_col)*TSC_weight(dist_row);
                acc_row_i += forces_y[cell_idx]*mass_inv * TSC_weight(dist_col)*TSC_weight(dist_row);
            
            }
        }
        particles->acc_col[i] = acc_col_i;
        particles->acc_row[i] = acc_row_i;
        
        // needed for the timestep selector
        particles->max_acc_col = MAX(particles->max_acc_col, fabs(acc_col_i));
        particles->max_acc_row = MAX(particles->max_acc_row, fabs(acc_row_i));
    }
}

void kick_particles(Particles* particles, double dt) {
    uint N_particles = particles->N;
    double* acc_col = particles->acc_col;
    double* acc_row = particles->acc_row;
    
    // auto vectorize 32, 16 byte vecs (versioned)
    for (uint i = 0; i < N_particles; i++) {
        particles->vel_col[i] += acc_col[i] * dt;
        particles->vel_row[i] += acc_row[i] * dt;
    }
}

void drift_particles(Particles* particles, double dt, vec2d_t BoxSize) {
    uint N_particles = particles->N;
    double* pos_col = particles->pos_col;
    double* pos_row = particles->pos_row;
    double* vel_col = particles->vel_col;
    double* vel_row = particles->vel_row;
    double ncols = BoxSize[_col_];
    double nrows = BoxSize[_row_];
    double pos_col_i, pos_row_i;
    // bool out_low, out_high;

    for (uint i = 0; i < N_particles; i++) {
        pos_col_i = pos_col[i] + vel_col[i] * dt;
        pos_row_i = pos_row[i] + vel_row[i] * dt;

        pos_col_i = fast_fmod(pos_col_i + ncols, ncols);
        pos_row_i = fast_fmod(pos_row_i + nrows, nrows);

        particles->pos_col[i] = pos_col_i;
        particles->pos_row[i] = pos_row_i;
    }
}
                