#include "init.h"
#include "io.h"
#include "pm.h"
#include "timing.h"
#include <stdio.h>

int main(int argc, char* argv[]) {

    // ----------------------------------------------------
    //                   INITIALIZATION                  //
    // ----------------------------------------------------

    if (argc != 2) {
        printf("Usage: %s params.conf\n", argv[0]);
        return EXIT_FAILURE;
    }

    // read parameters
    debug_print("Reading parameters from %s\n", argv[1]);
    Params* params = read_params(argv[1]);
    if (params == NULL) return EXIT_FAILURE;

    // compute G' normalized on parameters
    G_prime = compute_Gprime(&params->norm);

    // create the system
    debug_print("Initializing the particles\n");
    Particles* particles = init_particles(params);
    if (particles == NULL) return EXIT_FAILURE;

    // initialize the mesh
    debug_print("Initializing the mesh\n");
    Mesh* mesh = init_mesh(params->grid.Ngrid);
    if (mesh == NULL) return EXIT_FAILURE;

    // ----------------------------------------------------
    //                   PM PIPELINE                     //
    // ----------------------------------------------------

    uint grid_size = mesh->grid_size;
    double cell_size_col = params->grid.BoxSize[_col_] / params->grid.Ngrid[_col_];
    double cell_size_row = params->grid.BoxSize[_row_] / params->grid.Ngrid[_row_];

    double eta = 0.05;
    // auto vectorize 16 byte vecs
    double epsilon = MIN(cell_size_col, cell_size_row);

    // double half_cell_size_col = 0.5 * params->grid.BoxSize[_col_] / params->grid.Ngrid[_col_];
    // double half_cell_size_row = 0.5 * params->grid.BoxSize[_row_] / params->grid.Ngrid[_row_];

    double dt = 0.05; // initial timestep guess

    char filename[100];

    // auto vectorize 16 byte vecs
    vec2d_t BoxSize = {params->grid.BoxSize[_col_], params->grid.BoxSize[_row_]};
    // auto vectorize 8 byte vecs
    vec2_t Ngrid = {params->grid.Ngrid[_col_], params->grid.Ngrid[_row_]};

    debug_print("Running %d iterations\n", (int) params->system.n_iter);
    
    // timers
    double t_total_start = TCPU_TIME;
    double t_density = 0.0;
    double t_fft = 0.0;
    double t_potential = 0.0;
    double t_forces = 0.0;
    double t_interpolation = 0.0;
    double t_kick = 0.0;
    double t_drift = 0.0;
    double t0;
    
    for (uint t = 0; t < params->system.n_iter; t++) {
        debug_print("Iteration %d\n", (int) t);

        // 1. Estimate the density field on a grid
        debug_print("\tEstimating the density\n");
        t0 = TCPU_TIME;
        estimate_density(mesh, &params->grid, particles);
        t_density += TCPU_TIME - t0;
        if (mesh->density == NULL) return EXIT_FAILURE;

        // 2. Transform the density field to the Fourier space
        debug_print("\tFFT forward\n");
        t0 = TCPU_TIME;
        fftw_execute(mesh->fft_real_fwd);
        t_fft += TCPU_TIME - t0;

        // 3. Compute the gravitational potential using the green function of the Laplacian
        debug_print("\tComputing potential\n");
        t0 = TCPU_TIME;
        compute_potential(mesh, Ngrid, BoxSize);
        t_potential += TCPU_TIME - t0;
        
        // 4. Transform the gravitational potential back to the real space and evaluate the forces
        debug_print("\tFFT backward\n");
        t0 = TCPU_TIME;
        fftw_execute(mesh->fft_real_bck);
        t_fft += TCPU_TIME - t0;

        /* normalize */
        double norm = 1.0 / grid_size;

        // auto vectorize 32, 16 byte vecs
        for (uint i=0; i<grid_size; i++)
            mesh->pot[i] *= norm;
        
        // 5. Compute the forces on the grid
        debug_print("\tComputing forces on grid\n");
        t0 = TCPU_TIME;
        compute_forces(mesh, BoxSize, Ngrid);
        t_forces += TCPU_TIME - t0;

        // 6. Interpolate the forces to the particles
        debug_print("\tInterpolating forces to particles\n");
        t0 = TCPU_TIME;
        interpolate_forces(mesh, particles, BoxSize, Ngrid);
        t_interpolation += TCPU_TIME - t0;
        
        // 7. Update the particles positions and velocities using the forces and a leap-frog integrator
        debug_print("\tUpdating particles\n");
        t0 = TCPU_TIME;
        // half kick
        kick_particles(particles, dt/2.0);
        t_kick += TCPU_TIME - t0;
        // drift
        t0 = TCPU_TIME;
        drift_particles(particles, dt, BoxSize);
        t_drift += TCPU_TIME - t0;
        
        // recompute the forces
        t0 = TCPU_TIME;
        estimate_density(mesh, &params->grid, particles);
        t_density += TCPU_TIME - t0;
        if (mesh->density == NULL) return EXIT_FAILURE;
        
        t0 = TCPU_TIME;
        fftw_execute(mesh->fft_real_fwd);
        t_fft += TCPU_TIME - t0;
        
        t0 = TCPU_TIME;
        compute_potential(mesh, Ngrid, BoxSize);
        t_potential += TCPU_TIME - t0;
        
        t0 = TCPU_TIME;
        fftw_execute(mesh->fft_real_bck);
        t_fft += TCPU_TIME - t0;
        
        // auto vectorize 32, 16 byte vecs
        for (uint i=0; i<grid_size; i++) mesh->pot[i] *= norm;
        
        t0 = TCPU_TIME;
        compute_forces(mesh, BoxSize, Ngrid);
        t_forces += TCPU_TIME - t0;
        
        t0 = TCPU_TIME;
        interpolate_forces(mesh, particles, BoxSize, Ngrid);
        t_interpolation += TCPU_TIME - t0;

        // half kick
        t0 = TCPU_TIME;
        kick_particles(particles, dt/2.0);
        t_kick += TCPU_TIME - t0;

        // 8. Save the positions and status
        if (t % 3 == 0) {
            sprintf(filename, OUTPUT_POSITIONS_FILE, t, FORMAT);
            if (save_positions(particles, filename)) return EXIT_FAILURE;
            sprintf(filename, OUTPUT_STATUS_FILE, t, FORMAT);
            if (save_status(mesh, filename)) return EXIT_FAILURE;
        }

        // dt = sqrt(2 * eta * epsilon / max(abs(acc_col), abs(acc_row)))
        // auto vectorize 16 byte vecs
        double max_acc = MAX(particles->max_acc_col, particles->max_acc_row);
        if (max_acc > 0.0) {
            dt = sqrt(2 * eta * epsilon / max_acc);
        }
        
        // clamp the timestep
        if (dt < 1e-4) dt = 1e-4; // Lower bound
        if (dt > 0.5) dt = 0.5; // Upper bound
        debug_print("Timestep: %f\n", dt);
    }
    double t_total = TCPU_TIME - t_total_start;
    
    // Timing report (always printed for benchmarking)
    printf("\n=== TIMING REPORT ===\n");
    printf("Total loop time:     %10.6f s\n", t_total);
    printf("  estimate_density:  %10.6f s (%5.1f%%)\n", t_density, 100.0*t_density/t_total);
    printf("  FFT (fwd+bck):     %10.6f s (%5.1f%%)\n", t_fft, 100.0*t_fft/t_total);
    printf("  compute_potential: %10.6f s (%5.1f%%)\n", t_potential, 100.0*t_potential/t_total);
    printf("  compute_forces:    %10.6f s (%5.1f%%)\n", t_forces, 100.0*t_forces/t_total);
    printf("  interpolate:       %10.6f s (%5.1f%%)\n", t_interpolation, 100.0*t_interpolation/t_total);
    printf("  kick:              %10.6f s (%5.1f%%)\n", t_kick, 100.0*t_kick/t_total);
    printf("  drift:             %10.6f s (%5.1f%%)\n", t_drift, 100.0*t_drift/t_total);
    printf("======================\n\n");

    // save the final state of the particles
    sprintf(filename, OUTPUT_POSITIONS_FILE, params->system.n_iter, FORMAT);
    if (save_positions(particles, filename)) return EXIT_FAILURE;
    debug_print("Positions saved in %s\n", filename);
    
    sprintf(filename, OUTPUT_STATUS_FILE, params->system.n_iter, FORMAT);
    if (save_status(mesh, filename)) return EXIT_FAILURE;
    debug_print("Status saved in %s\n", filename);

    // destroy the system
    destroy_particles(particles);
    destroy_mesh(mesh);
    free(params);
    debug_print("System destroyed\n");

    return EXIT_SUCCESS;
}