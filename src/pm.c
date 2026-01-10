#include "pm.h"

double NGP_weight(double dist) {
    if (dist < 0.5)
        return 1.0;
    return 0.0;
}

double CIC_weight(double dist) {
    // 1 - |x|
    if (dist < 1.0)
        return 1.0 - dist;
    // else 0.0
    return 0.0;
}

double TSC_weight(double dist) {
    // 3/4 - |x|^2
    if (dist < 0.5)
        return 0.75 - dist * dist;
    // 1/2 (3/2 - |x|)^2
    else if ((dist > 0.5) && (dist < 1.5))
        return 0.5 * (1.5 - dist) * (1.5 - dist);
    // else 0.0
    return 0.0;
}

int get_cell_index(double grid_idx, uint N_grid) {
    double size = (double) N_grid;
    if (grid_idx < 0.0 || grid_idx >= size) {
        if (!PERIODIC) {
            return -1;
        } else {
            return (int)fmod(grid_idx + size, size);
        }
    }
    return (int) grid_idx;
}

double* estimate_density_NGP(Mesh* mesh, GridParams* params, Particles* particles) {
    // extract grid parameters for efficiency
    uint N_cols = params->Ngrid[_col_];
    uint N_rows = params->Ngrid[_row_];
    uint N_particles = particles->N;
    double cellSize_col = params->BoxSize[_col_] / N_cols;
    double cellSize_row = params->BoxSize[_row_] / N_rows;

    // initialize density array
    memset(mesh->density, 0, N_cols * N_rows * sizeof(double));

    // iterate over particles
    for (uint i = 0; i < N_particles; i++) {

        // get particle position in grid units
        double grid_col = particles->pos_col[i] / cellSize_col;
        double grid_row = particles->pos_row[i] / cellSize_row;

        
        // get absolute relative position and distance from the grid cell center
        double dist_col = fabs(grid_col - (floor(grid_col) + 0.5));
        double dist_row = fabs(grid_row - (floor(grid_row) + 0.5));

        
        // update density
        // row-major layout: contiguous cells advance along x, then y
        int col_idx = get_cell_index(floor(grid_col), N_cols);
        int row_idx = get_cell_index(floor(grid_row), N_rows);
        // skip if out of bounds and not periodic
        if (col_idx == -1 || row_idx == -1) continue;
        int grid_idx = row_idx * N_cols + col_idx;
        mesh->density[grid_idx] += particles->mass[i] * NGP_weight(dist_col)*NGP_weight(dist_row);
    }

    // return new density array
    return mesh->density;
}

double* estimate_density_CIC(Mesh* mesh, GridParams* params, Particles* particles) {
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = params->Ngrid[_col_];
    uint N_rows = params->Ngrid[_row_];
    double cellSize_col = params->BoxSize[_col_] / N_cols;
    double cellSize_row = params->BoxSize[_row_] / N_rows;

    // initialize density array
    memset(mesh->density, 0, N_cols * N_rows * sizeof(double));

    // iterate over particles
    for (uint i = 0; i < N_particles; i++) {

        // get particle position in grid units
        double grid_col = particles->pos_col[i] / cellSize_col;
        double grid_row = particles->pos_row[i] / cellSize_row;

        // choose the two nearest cell centers along each axis:
        // if the particle lies left of the cell center, stencil spans (-1, 0),
        // otherwise it spans (0, +1).
        int col_start = (grid_col < (floor(grid_col) + 0.5)) ? -1 : 0;
        int row_start = (grid_row < (floor(grid_row) + 0.5)) ? -1 : 0;
        int col_end = col_start + 1;
        int row_end = row_start + 1;

        for (int dc = col_start; dc <= col_end; dc++) {
            for (int dr = row_start; dr <= row_end; dr++) {

                double col_neighbor = floor(grid_col) + dc;
                double row_neighbor = floor(grid_row) + dr;

                // get relative position and distance from the grid cell center
                double dist_col = fabs(grid_col - (col_neighbor + 0.5));
                double dist_row = fabs(grid_row - (row_neighbor + 0.5));

                if (dist_col >= 1.0 || dist_row >= 1.0) continue;

                int col_idx = get_cell_index(col_neighbor, N_cols);
                int row_idx = get_cell_index(row_neighbor, N_rows);
                // skip if out of bounds and not periodic
                if (col_idx == -1 || row_idx == -1) continue;
                
                // row-major layout: y-major rows, x-major columns
                int cell_idx = row_idx * N_cols + col_idx;
                
                mesh->density[cell_idx] += particles->mass[i] * CIC_weight(dist_col)*CIC_weight(dist_row);
            }
        }
    }

    // return new density array
    return mesh->density;
}

double* estimate_density_TSC(Mesh* mesh, GridParams* params, Particles* particles) {
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = params->Ngrid[_col_];
    uint N_rows = params->Ngrid[_row_];
    double cellSize_col = params->BoxSize[_col_] / N_cols;
    double cellSize_row = params->BoxSize[_row_] / N_rows;

    // initialize density array
    memset(mesh->density, 0, N_cols * N_rows * sizeof(double));

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
            for (int r = row_min; r <= row_max; r++) {

                double col_neighbor = (double)c;
                double row_neighbor = (double)r;

                // get relative position and distance from the grid cell center
                double dist_col = fabs(grid_col - (col_neighbor + 0.5));
                double dist_row = fabs(grid_row - (row_neighbor + 0.5));

                // skip if outside TSC support (distance > 1.5)
                if (dist_col >= 1.5 || dist_row >= 1.5) continue;

                int col_idx = get_cell_index(col_neighbor, N_cols);
                int row_idx = get_cell_index(row_neighbor, N_rows);
                // skip if out of bounds and not periodic
                if (col_idx == -1 || row_idx == -1) continue;
                
                // row-major layout: y-major rows, x-major columns
                int cell_idx = row_idx * N_cols + col_idx;
                
                mesh->density[cell_idx] += particles->mass[i] * TSC_weight(dist_col)*TSC_weight(dist_row);
            }
        }
    }

    // return new density array
    return mesh->density;
}

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
            if (denom <= 1e-14) {
                // zero mode: set to 0 to avoid division by zero and unphysical drift
                green_func = 0.0;
            } else {
                green_func = -(4 * M_PI * G_prime) / denom;
            }

            mesh->kPot[idx][0] = green_func * mesh->kDensity[idx][0];
            mesh->kPot[idx][1] = green_func * mesh->kDensity[idx][1];
        }
    }
}

void compute_forces(Mesh* mesh, vec2d_t BoxSize, vec2_t Ngrid) {
    uint prev_col, next_col, prev_row, next_row, idx, i, j;
    double* pot = mesh->pot;

    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];

    double cellSize_col = BoxSize[_col_] / N_cols;
    double cellSize_row = BoxSize[_row_] / N_rows;

    double den_col = 2 * cellSize_col;
    double den_row = 2 * cellSize_row;

    for (i = 0; i < N_rows; i++) {
        for (j = 0; j < N_cols; j++) {
            idx = i * N_cols + j;

            #ifndef NPERIODIC
                prev_col = (j - 1 + N_cols) % N_cols;
                next_col = (j + 1) % N_cols;
                prev_row = (i - 1 + N_rows) % N_rows;
                next_row = (i + 1) % N_rows;

                // central difference approximation of the gradient
                mesh->forces_x[idx] = (pot[i * N_cols + prev_col] - pot[i * N_cols + next_col]) / den_col;
                mesh->forces_y[idx] = (pot[prev_row * N_cols + j] - pot[next_row * N_cols + j]) / den_row;
            #else
                // Non-periodic fallback: one-sided differences at boundaries
                if (j == 0) {
                    mesh->forces_x[idx] = (pot[i * N_cols + j] - pot[i * N_cols + (j + 1)]) / cellSize_col;
                } else if (j == N_cols - 1) {
                    mesh->forces_x[idx] = (pot[i * N_cols + (j - 1)] - pot[i * N_cols + j]) / cellSize_col;
                } else {
                    mesh->forces_x[idx] = (pot[i * N_cols + (j - 1)] - pot[i * N_cols + (j + 1)]) / den_col;
                }

                if (i == 0) {
                    mesh->forces_y[idx] = (pot[i * N_cols + j] - pot[(i + 1) * N_cols + j]) / cellSize_row;
                } else if (i == N_rows - 1) {
                    mesh->forces_y[idx] = (pot[(i - 1) * N_cols + j] - pot[i * N_cols + j]) / cellSize_row;
                } else {
                    mesh->forces_y[idx] = (pot[(i - 1) * N_cols + j] - pot[(i + 1) * N_cols + j]) / den_row;
                }
            #endif
        }
    }
}

void interpolate_forces_NGP(Mesh* mesh, Particles* particles, vec2d_t BoxSize, vec2_t Ngrid) {
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];
    double cellSize_col = BoxSize[_col_] / N_cols;
    double cellSize_row = BoxSize[_row_] / N_rows;
    double* forces_x = mesh->forces_x;
    double* forces_y = mesh->forces_y;
    double* mass = particles->mass;

    // initialize the forces on the particles
    memset(particles->acc_col, 0, N_particles * sizeof(double));
    memset(particles->acc_row, 0, N_particles * sizeof(double));
    particles->max_acc_col = 0.0;
    particles->max_acc_row = 0.0;
    
    // iterate over particles
    for (uint i = 0; i < N_particles; i++) {

        // get particle position in grid units
        double grid_col = particles->pos_col[i] / cellSize_col;
        double grid_row = particles->pos_row[i] / cellSize_row;

        // get absolute relative position and distance from the grid cell center
        double dist_col = fabs(grid_col - (floor(grid_col) + 0.5));
        double dist_row = fabs(grid_row - (floor(grid_row) + 0.5));

        // update the forces on the particles
        // row-major layout: contiguous cells advance along x, then y
        int col_idx = get_cell_index(floor(grid_col), N_cols);
        int row_idx = get_cell_index(floor(grid_row), N_rows);
        // skip if out of bounds and not periodic
        if (col_idx == -1 || row_idx == -1) continue;
        int grid_idx = row_idx * N_cols + col_idx;

        particles->acc_col[i] += forces_x[grid_idx]/mass[i] * NGP_weight(dist_col)*NGP_weight(dist_row);
        particles->acc_row[i] += forces_y[grid_idx]/mass[i] * NGP_weight(dist_col)*NGP_weight(dist_row);
    
        // needed for the timestep selector
        particles->max_acc_col = MAX(particles->max_acc_col, fabs(particles->acc_col[i]));
        particles->max_acc_row = MAX(particles->max_acc_row, fabs(particles->acc_row[i]));
    }
}

void interpolate_forces_CIC(Mesh* mesh, Particles* particles, vec2d_t BoxSize, vec2_t Ngrid) {
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];
    double cellSize_col = BoxSize[_col_] / N_cols;
    double cellSize_row = BoxSize[_row_] / N_rows;
    double* forces_x = mesh->forces_x;
    double* forces_y = mesh->forces_y;
    double* mass = particles->mass;

    // initialize the forces on the particles
    memset(particles->acc_col, 0, N_particles * sizeof(double));
    memset(particles->acc_row, 0, N_particles * sizeof(double));
    particles->max_acc_col = 0.0;
    particles->max_acc_row = 0.0;

    // iterate over particles
    for (uint i = 0; i < N_particles; i++) {

        // get particle position in grid units
        double grid_col = particles->pos_col[i] / cellSize_col;
        double grid_row = particles->pos_row[i] / cellSize_row;

        // choose the two nearest cell centers along each axis:
        // if the particle lies left of the cell center, stencil spans (-1, 0),
        // otherwise it spans (0, +1).
        int col_start = (grid_col < (floor(grid_col) + 0.5)) ? -1 : 0;
        int row_start = (grid_row < (floor(grid_row) + 0.5)) ? -1 : 0;
        int col_end = col_start + 1;
        int row_end = row_start + 1;

        for (int dc = col_start; dc <= col_end; dc++) {
            for (int dr = row_start; dr <= row_end; dr++) {

                double col_neighbor = floor(grid_col) + dc;
                double row_neighbor = floor(grid_row) + dr;

                // get relative position and distance from the grid cell center
                double dist_col = fabs(grid_col - (col_neighbor + 0.5));
                double dist_row = fabs(grid_row - (row_neighbor + 0.5));

                if (dist_col >= 1.0 || dist_row >= 1.0) continue;

                int col_idx = get_cell_index(col_neighbor, N_cols);
                int row_idx = get_cell_index(row_neighbor, N_rows);
                // skip if out of bounds and not periodic
                if (col_idx == -1 || row_idx == -1) continue;
                
                // row-major layout: y-major rows, x-major columns
                int cell_idx = row_idx * N_cols + col_idx;

                particles->acc_col[i] += forces_x[cell_idx]/mass[i] * CIC_weight(dist_col)*CIC_weight(dist_row);
                particles->acc_row[i] += forces_y[cell_idx]/mass[i] * CIC_weight(dist_col)*CIC_weight(dist_row);
                
                // needed for the timestep selector
                particles->max_acc_col = MAX(particles->max_acc_col, fabs(particles->acc_col[i]));
                particles->max_acc_row = MAX(particles->max_acc_row, fabs(particles->acc_row[i]));
            }
        }
    }
}

void interpolate_forces_TSC(Mesh* mesh, Particles* particles, vec2d_t BoxSize, vec2_t Ngrid) {
    
    // extract grid parameters for efficiency
    uint N_particles = particles->N;
    uint N_cols = Ngrid[_col_];
    uint N_rows = Ngrid[_row_];
    double cellSize_col = BoxSize[_col_] / N_cols;
    double cellSize_row = BoxSize[_row_] / N_rows;
    double* forces_x = mesh->forces_x;
    double* forces_y = mesh->forces_y;
    double* mass = particles->mass;
    
    // initialize the forces on the particles
    memset(particles->acc_col, 0, N_particles * sizeof(double));
    memset(particles->acc_row, 0, N_particles * sizeof(double));
    particles->max_acc_col = 0.0;
    particles->max_acc_row = 0.0;

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
            for (int r = row_min; r <= row_max; r++) {

                double col_neighbor = (double)c;
                double row_neighbor = (double)r;

                // get relative position and distance from the grid cell center
                double dist_col = fabs(grid_col - (col_neighbor + 0.5));
                double dist_row = fabs(grid_row - (row_neighbor + 0.5));

                // skip if outside TSC support (distance > 1.5)
                if (dist_col >= 1.5 || dist_row >= 1.5) continue;

                int col_idx = get_cell_index(col_neighbor, N_cols);
                int row_idx = get_cell_index(row_neighbor, N_rows);
                // skip if out of bounds and not periodic
                if (col_idx == -1 || row_idx == -1) continue;
                
                // row-major layout: y-major rows, x-major columns
                int cell_idx = row_idx * N_cols + col_idx;
                
                particles->acc_col[i] += forces_x[cell_idx]/mass[i] * TSC_weight(dist_col)*TSC_weight(dist_row);
                particles->acc_row[i] += forces_y[cell_idx]/mass[i] * TSC_weight(dist_col)*TSC_weight(dist_row);
            
                // needed for the timestep selector
                particles->max_acc_col = MAX(particles->max_acc_col, fabs(particles->acc_col[i]));
                particles->max_acc_row = MAX(particles->max_acc_row, fabs(particles->acc_row[i]));
            }
        }
    }
}

void kick_particles(Particles* particles, double dt) {
    uint N_particles = particles->N;
    double* acc_col = particles->acc_col;
    double* acc_row = particles->acc_row;

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

    for (uint i = 0; i < N_particles; i++) {
        pos_col_i = pos_col[i] + vel_col[i] * dt;
        pos_row_i = pos_row[i] + vel_row[i] * dt;

        // if periodic, wrap the positions around the edges of the box
        #ifndef NPERIODIC
            pos_col_i = fmod(pos_col_i + ncols, ncols);
            pos_row_i = fmod(pos_row_i + nrows, nrows);
        #endif

        particles->pos_col[i] = pos_col_i;
        particles->pos_row[i] = pos_row_i;
    }
}
                