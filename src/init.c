#include "init.h"

// global normalized gravitational constant
real_t G_prime = 0.0;

Params* read_params(int argc, char* argv[]) {

    char* filename = argv[1];
    if (filename == NULL) {
        printf("Error: missing parameters file\n");
        return NULL;
    }
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("fopen");
        return NULL;
    }

    Params* params = (Params*) malloc(sizeof(Params));
    if (!params) {
        printf("Error while allocating params");
        fclose(file);
        return NULL;
    }
    memset(params, 0, sizeof(*params));

    char* lineptr = NULL;
    size_t n = 0;

    char name[32];
    real_t value;

    while ((getline(&lineptr, &n, file)) != -1) {
        // Skip comments and blank lines
        if (lineptr[0] == '#' || lineptr[0] == '\n')
            continue;

        // Try to parse the line
        if (sscanf(lineptr, " %31[^= \t\r\n] = " REAL_FMT, name, &value) == 2) {
            
            // Normalization params
            if (strcmp(name, "UnitVel") == 0) {
                params->norm.UnitVel = value;
            } else if (strcmp(name, "UnitMass") == 0) {
                params->norm.UnitMass = value;
            } else if (strcmp(name, "UnitLength") == 0) {
                params->norm.UnitLength = value;
            } else if (strcmp(name, "UnitTime") == 0) {
                params->norm.UnitTime = value;
            } 

            // Grid params
            else if (strcmp(name, "Npoints") == 0) {
                params->grid.Npoints = (uint) value;
            } else if (strcmp(name, "NgridX") == 0) {
                params->grid.Ngrid[0] = (uint) value;
            } else if (strcmp(name, "NgridY") == 0) {
                params->grid.Ngrid[1] = (uint) value;
            } else if (strcmp(name, "BoxSizeX") == 0) {
                params->grid.BoxSize[0] = value;
            } else if (strcmp(name, "BoxSizeY") == 0) {
                params->grid.BoxSize[1] = value;
            } 

            // System params
            else if (strcmp(name, "A_deltaPar") == 0) {
                params->system.A_deltaPar = value;
            } else if (strcmp(name, "n_iter") == 0) {
                params->system.n_iter = (uint) value;
            }
        }
    }
    
    free(lineptr);
    fclose(file);

    debug_print("UnitVel \t= %g\n", params->norm.UnitVel);
    debug_print("UnitMass \t= %g\n", params->norm.UnitMass);
    debug_print("UnitLength \t= %g\n", params->norm.UnitLength);
    debug_print("UnitTime \t= %g\n", params->norm.UnitTime);
    debug_print("Npoints \t= %d\n", params->grid.Npoints);
    debug_print("NgridX  \t= %d\n", params->grid.Ngrid[0]);
    debug_print("NgridY  \t= %d\n", params->grid.Ngrid[1]);
    debug_print("BoxSizeX \t= %g\n", params->grid.BoxSize[0]);
    debug_print("BoxSizeY \t= %g\n", params->grid.BoxSize[1]);
    debug_print("A_deltaPar\t= %g\n", params->system.A_deltaPar);

    if (argc > 2) { // args: params_file [Npoints] [grid_x] [grid_y]
        if (argc > 2) params->grid.Npoints = (uint) atoi(argv[2]);
        if (argc > 3) params->grid.Ngrid[0] = (uint) atoi(argv[3]);
        if (argc > 4) params->grid.Ngrid[1] = (uint) atoi(argv[4]);
        if (argc > 5) params->system.n_iter = (uint) atoi(argv[5]);
    }

    if ((!params->norm.UnitVel || !params->norm.UnitMass || !params->norm.UnitLength || !params->norm.UnitTime) ||
        (!params->grid.Npoints || !params->grid.Ngrid[0] || !params->grid.Ngrid[1]) ||
        (!params->grid.BoxSize[0] || !params->grid.BoxSize[1]) ||
        (!params->system.A_deltaPar)) {
        printf("Error: missing parameters\n");
        free(params);
        return NULL;
    }

    return params;
}

// Initialize a plane (2D array)
Particles* init_particles(Params* params) {
    // Allocate particles to size Npoints
    Particles* particles = (Particles*) malloc(sizeof(Particles));
    if (particles == NULL) {
        printf("Error while allocating memory for the particles\n");
        return NULL;
    }
    
    memset(particles, 0, sizeof(Particles));

    int N = params->grid.Npoints;
    if (RANDOM) { // round up to the nearest square number
        N = (int) SQRT((real_t) N);
        N = N * N;
    }
    particles->N = N;

    particles->pos_col = (real_t*) allocate_aligned(particles->N * sizeof(real_t));
    particles->pos_row = (real_t*) allocate_aligned(particles->N * sizeof(real_t));
    if (particles->pos_col == NULL || particles->pos_row == NULL) {
        printf("Error while allocating memory for the particles positions\n");
        destroy_particles(particles);
        return NULL;
    }

    particles->vel_col = (real_t*) allocate_aligned(particles->N * sizeof(real_t));
    particles->vel_row = (real_t*) allocate_aligned(particles->N * sizeof(real_t));
    if (particles->vel_col == NULL || particles->vel_row == NULL) {
        printf("Error while allocating memory for the particles velocities\n");
        destroy_particles(particles);
        return NULL;
    }

    particles->mass = (real_t)0.01;

    particles->acc_col = (real_t*) allocate_aligned(particles->N * sizeof(real_t));
    if (particles->acc_col == NULL) {
        printf("Error while allocating memory for the particles forces x\n");
        destroy_particles(particles);
        return NULL;
    }

    particles->acc_row = (real_t*) allocate_aligned(particles->N * sizeof(real_t));
    if (particles->acc_row == NULL) {
        printf("Error while allocating memory for the particles forces y\n");
        destroy_particles(particles);
        return NULL;
    }

    #ifdef USE_OMP
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < particles->N; i++) {
        particles->vel_col[i] = 0.0;
        particles->vel_row[i] = 0.0;
        particles->acc_col[i] = 0.0;
        particles->acc_row[i] = 0.0;
    }
    #else
    memset(particles->vel_col, 0, particles->N * sizeof(real_t));
    memset(particles->vel_row, 0, particles->N * sizeof(real_t));
    memset(particles->acc_col, 0, particles->N * sizeof(real_t));
    memset(particles->acc_row, 0, particles->N * sizeof(real_t));
    #endif

    // auto vectorize 16 byte vecs
    particles->max_acc_col = 0.0;
    particles->max_acc_row = 0.0;

    // Init particles
    place_particles(particles, params->grid.BoxSize, params->system.A_deltaPar);

    // Initialization successful
    return particles;
}

// G' normalized on parameters
real_t compute_Gprime(NormalizationParams* norm) {
    return G_SI * norm->UnitMass * (norm->UnitTime * norm->UnitTime) / (norm->UnitLength * norm->UnitLength * norm->UnitLength);
}

void place_particles(Particles* particles, vec2d_t BoxSize, real_t A_deltaPar) {
    srand48(42); // Fixed seed for reproducibility

    int N = particles->N;
    // If N is not a perfect square, still create a near-square lattice and fill row-major.
    int ncols = (int)CEIL(SQRT((real_t)N));
    int nrows = (int)CEIL((real_t)N / (real_t)ncols);
    real_t dx_grid = BoxSize[_col_] / (real_t)ncols;
    real_t dy_grid = BoxSize[_row_] / (real_t)nrows;

    real_t k_col = 2 * M_PI_T / BoxSize[_col_];
    real_t k_row = 2 * M_PI_T / BoxSize[_row_];
    real_t half_pi = M_PI_T/2;

    // Simple 2D extension of 1D Zel'dovich scaling: displacement amplitude ~ A_deltaPar / k.
    real_t psi_amp_col = (FABS(k_col) > 0.0) ? (A_deltaPar / k_col) : 0.0;
    real_t psi_amp_row = (FABS(k_row) > 0.0) ? (A_deltaPar / k_row) : 0.0;

    for (int i = 0; i < N; i++) {
        real_t q_col, q_row;

        if (RANDOM) {
            q_col = drand48() * BoxSize[_col_];
            q_row = drand48() * BoxSize[_row_];
        }
        // if not random, place the particles on a homogeneous grid
        else {
            int col_idx = ((int) i) % ncols;
            int row_idx = ((int) i) / ncols;
            q_col = (col_idx + 0.5) * dx_grid;
            q_row = (row_idx + 0.5) * dy_grid;
        }

        /* Zeldovich Approximation to create a density contrast
        *  delta(x) = A * sin((2*pi*x)/L + pi/2)
        *  shift psi(q) ~ - int(delta(q)) = A/k * cos(k q + pi/2),    where k = 2*pi/L
        */

        real_t psi_col = psi_amp_col * COS(k_col * q_col - half_pi);
        real_t psi_row = psi_amp_row * COS(k_row * q_row - half_pi);

        particles->pos_col[i] = q_col + psi_col;
        particles->pos_row[i] = q_row + psi_row;
    }
}

Mesh* init_mesh(vec2_t Ngrid) {

    FFTW_INIT_THREADS;
    FFTW_PLAN_WITH_NTHREADS;

    Mesh* mesh = (Mesh*) malloc(sizeof(Mesh));
    if (mesh == NULL) {
        printf("Error while allocating memory for the mesh\n");
        return NULL;
    }

    // FFTW r2c_2d(n0, n1, ...) output is sized: n0 * (n1/2 + 1)
    // Here: n0 = Ngrid[_row_], n1 = Ngrid[_col_]
    size_t size     = (size_t) Ngrid[_row_] * Ngrid[_col_];
    size_t fft_size = (size_t) Ngrid[_row_] * (Ngrid[_col_] / 2 + 1);
    mesh->grid_size = size;

    mesh->kDensity = (complex_t*) pm_malloc(fft_size, sizeof(complex_t));
    mesh->kPot     = (complex_t*) pm_malloc(fft_size, sizeof(complex_t));

    // allocate memory for the real-space arrays
    mesh->density  = (real_t*) allocate_aligned(size * sizeof(real_t));
    mesh->pot      = (real_t*) allocate_aligned(size * sizeof(real_t));

    // allocate memory for the forces arrays
    mesh->forces_x = (real_t*) allocate_aligned(size * sizeof(real_t));
    mesh->forces_y = (real_t*) allocate_aligned(size * sizeof(real_t));

    if (!mesh->kDensity || !mesh->kPot || !mesh->density || !mesh->pot) {
        printf("Error while allocating mesh arrays\n");
        destroy_mesh(mesh);
        return NULL;
    }

    // zero-initialize real-space arrays for deterministic transforms
    #ifdef USE_OMP
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < size; i++) {
        mesh->density[i] = 0.0;
        mesh->pot[i] = 0.0;
    }
    #else
    memset(mesh->density, 0, size * sizeof(real_t));
    memset(mesh->pot, 0, size * sizeof(real_t));
    #endif

    // back and forth FFT plans (note: FFTW takes int dimensions)
    #ifdef USE_GPU
        if (cufftPlan2d(&mesh->plan_fwd, Ngrid[_row_], Ngrid[_col_], CUFFT_TYPE_R2C) != CUFFT_SUCCESS) {
             printf("Error creating cuFFT FWD plan\n"); return NULL;
        }
        if (cufftPlan2d(&mesh->plan_bck, Ngrid[_row_], Ngrid[_col_], CUFFT_TYPE_C2R) != CUFFT_SUCCESS) {
             printf("Error creating cuFFT BCK plan\n"); return NULL;
        }
    #else
    mesh->plan_fwd = FFTW_PLAN_DFT_R2C_2D((int) Ngrid[_row_], (int) Ngrid[_col_], mesh->density, mesh->kDensity, FFTW_MEASURE);
    mesh->plan_bck = FFTW_PLAN_DFT_C2R_2D((int) Ngrid[_row_], (int) Ngrid[_col_], mesh->kPot, mesh->pot, FFTW_MEASURE);
    if (!mesh->plan_fwd || !mesh->plan_bck) {
        printf("Error while creating FFTW plans\n");
        destroy_mesh(mesh);
        return NULL;
    }
    #endif

    return mesh;
}

int destroy_mesh(Mesh* mesh) {
    FFTW_CLEANUP_THREADS;
    if (mesh->plan_fwd) PM_DESTROY_PLAN(mesh->plan_fwd);
    if (mesh->plan_bck) PM_DESTROY_PLAN(mesh->plan_bck);
    if (mesh->kDensity) pm_free(mesh->kDensity);
    if (mesh->kPot) pm_free(mesh->kPot);
    if (mesh->density) free(mesh->density);
    if (mesh->pot) free(mesh->pot);
    if (mesh->forces_x) free(mesh->forces_x);
    if (mesh->forces_y) free(mesh->forces_y);
    if (mesh) free(mesh);
    return EXIT_SUCCESS;
}

int destroy_particles(Particles* particles) {
    if (particles->pos_col) free(particles->pos_col);
    if (particles->pos_row) free(particles->pos_row);
    if (particles->vel_col) free(particles->vel_col);
    if (particles->vel_row) free(particles->vel_row);
    if (particles->acc_col) free(particles->acc_col);
    if (particles->acc_row) free(particles->acc_row);
    if (particles) free(particles);
    return EXIT_SUCCESS;
}
