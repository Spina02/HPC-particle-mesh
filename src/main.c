#include "init.h"
#include "io.h"
#include "pm.h"
#include "timing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {

    // ----------------------------------------------------
    //                   INITIALIZATION                  //
    // ----------------------------------------------------

    if (argc < 2) {
        printf("Usage: %s params.conf\n", argv[0]);
        return EXIT_FAILURE;
    }

    // read parameters
    debug_print("Reading parameters from %s\n", argv[1]);
    Params* params = read_params(argc, argv);
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
    
    uint grid_size = mesh->grid_size;
    double cell_size_col = params->grid.BoxSize[_col_] / params->grid.Ngrid[_col_];
    double cell_size_row = params->grid.BoxSize[_row_] / params->grid.Ngrid[_row_];
    
    double norm = 1.0 / grid_size;
    double eta = 0.05;
    double epsilon = MIN(cell_size_col, cell_size_row);
    double two_eta_epsilon = 2.0 * eta * epsilon;
    double dt = 0.05; // initial timestep guess
    
    // auto vectorize 16 byte vecs
    vec2d_t BoxSize = {params->grid.BoxSize[_col_], params->grid.BoxSize[_row_]};
    // auto vectorize 8 byte vecs
    vec2_t Ngrid = {params->grid.Ngrid[_col_], params->grid.Ngrid[_row_]};
    
    // reorder the particles for better cache locality
    reorder_particles(particles, &params->grid);
    
    // save the initial state of the particles
    #ifdef SAVE
        char filename[100];
        sprintf(filename, OUTPUT_POSITIONS_FILE, 0, FORMAT);
        if (save_positions(particles, filename)) return EXIT_FAILURE;
        sprintf(filename, OUTPUT_STATUS_FILE, 0, FORMAT);   
        if (save_status(mesh, filename)) return EXIT_FAILURE;
    #endif
    
    // timers
    double t_total_start = TCPU_TIME;
    double t_density = 0.0;
    double t_fft = 0.0;
    double t_potential = 0.0;
    double t_forces = 0.0;
    double t_interpolation = 0.0;
    double t_kick = 0.0;
    double t_drift = 0.0;
    double t_reorder = 0.0;
    double t0;

    // ----------------------------------------------------
    //                   PM PIPELINE                     //
    // ----------------------------------------------------
    
    for (uint t = 0; t < params->system.n_iter; t++) {

        debug_print("Iteration %d\n", (int) t);

        // reorder the particles every 10 iterations for better cache locality
        if ((t + 1) % 20 == 0) {
            t0 = TCPU_TIME;
            reorder_particles(particles, &params->grid);
            t_reorder += TCPU_TIME - t0;
        }

        // 1. Estimate the density field on a grid
        debug_print("\tEstimating the density\n");
        t0 = TCPU_TIME;
        estimate_density(mesh, &params->grid, particles);
        t_density += TCPU_TIME - t0;

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

        // normalize the potential
        #if defined(OMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (uint i=0; i<grid_size; i++) mesh->pot[i] *= norm;
        
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
        
        t0 = TCPU_TIME;
        fftw_execute(mesh->fft_real_fwd);
        t_fft += TCPU_TIME - t0;
        
        t0 = TCPU_TIME;
        compute_potential(mesh, Ngrid, BoxSize);
        t_potential += TCPU_TIME - t0;
        
        t0 = TCPU_TIME;
        fftw_execute(mesh->fft_real_bck);
        t_fft += TCPU_TIME - t0;
        
        #if defined(OMP)
        #pragma omp parallel for schedule(static)
        #endif
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
        #ifdef SAVE
        if (t % 3 == 0) {
            sprintf(filename, OUTPUT_POSITIONS_FILE, t, FORMAT);
            if (save_positions(particles, filename)) return EXIT_FAILURE;
            sprintf(filename, OUTPUT_STATUS_FILE, t, FORMAT);
            if (save_status(mesh, filename)) return EXIT_FAILURE;
        }
        #endif

        // dt = sqrt(2 * eta * epsilon / max(abs(acc_col), abs(acc_row)))
        // auto vectorize 16 byte vecs
        double max_acc = MAX(particles->max_acc_col, particles->max_acc_row);
        if (max_acc > 0.0) {
            dt = sqrt(two_eta_epsilon / max_acc);
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
    printf("  reorder:           %10.6f s (%5.1f%%)\n", t_reorder, 100.0*t_reorder/t_total);
    printf("======================\n\n");

    // Write timing to CSV if TIMING_CSV environment variable is set
    char* csv_path = getenv("TIMING_CSV");
    if (csv_path != NULL) {
        // Get SLURM environment variables (default to 1 if not set)
        int nodes = 1, tasks_per_node = 1, cpus_per_task = 1;
        char* env_val;
        if ((env_val = getenv("SLURM_NNODES")) != NULL) nodes = atoi(env_val);
        if ((env_val = getenv("SLURM_NTASKS_PER_NODE")) != NULL) tasks_per_node = atoi(env_val);
        if ((env_val = getenv("SLURM_CPUS_PER_TASK")) != NULL) cpus_per_task = atoi(env_val);

        // Check if file exists to write header
        FILE* csv_check = fopen(csv_path, "r");
        int write_header = (csv_check == NULL);
        if (csv_check != NULL) fclose(csv_check);

        // Open file in append mode
        FILE* csv = fopen(csv_path, "a");
        if (csv != NULL) {
            if (write_header) {
                fprintf(csv, "nodes,tasks_per_node,cpus_per_task,npoints,ngrid_x,ngrid_y,total,density,fft,potential,forces,interpolation,kick,drift,reorder\n");
            }
            fprintf(csv, "%d,%d,%d,%u,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    nodes, tasks_per_node, cpus_per_task,
                    params->grid.Npoints, params->grid.Ngrid[0], params->grid.Ngrid[1],
                    t_total, t_density, t_fft, t_potential, t_forces, t_interpolation, t_kick, t_drift, t_reorder);
            fclose(csv);
            printf("Timing data appended to %s\n", csv_path);
        } else {
            fprintf(stderr, "Warning: Could not open CSV file %s for writing\n", csv_path);
        }
    }

    #ifdef SAVE
        // save the final state of the particles
        sprintf(filename, OUTPUT_POSITIONS_FILE, params->system.n_iter, FORMAT);
        if (save_positions(particles, filename)) return EXIT_FAILURE;
        debug_print("Positions saved in %s\n", filename);
        
        sprintf(filename, OUTPUT_STATUS_FILE, params->system.n_iter, FORMAT);
        if (save_status(mesh, filename)) return EXIT_FAILURE;
        debug_print("Status saved in %s\n", filename);
    #endif

    // destroy the system
    destroy_particles(particles);
    destroy_mesh(mesh);
    free(params);
    debug_print("System destroyed\n");

    return EXIT_SUCCESS;
}