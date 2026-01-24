#include "init.h"

// global normalized gravitational constant
double G_prime = 0.0;

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
    ssize_t nread;

    char name[32];
    double value;

    while ((nread = getline(&lineptr, &n, file)) != -1) {
        // Skip comments and blank lines
        if (lineptr[0] == '#' || lineptr[0] == '\n')
            continue;

        // Try to parse the line
        if (sscanf(lineptr, " %31[^= \t\r\n] = %lf", name, &value) == 2) {
            
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
        N = (int) sqrt((double) N);
        N = N * N;
    }
    particles->N = N;

    particles->pos_col = (double*) allocate_aligned(particles->N * sizeof(double));
    particles->pos_row = (double*) allocate_aligned(particles->N * sizeof(double));
    if (particles->pos_col == NULL || particles->pos_row == NULL) {
        printf("Error while allocating memory for the particles positions\n");
        destroy_particles(particles);
        return NULL;
    }

    particles->vel_col = (double*) allocate_aligned(particles->N * sizeof(double));
    particles->vel_row = (double*) allocate_aligned(particles->N * sizeof(double));
    if (particles->vel_col == NULL || particles->vel_row == NULL) {
        printf("Error while allocating memory for the particles velocities\n");
        destroy_particles(particles);
        return NULL;
    }

    particles->mass = 0.01;

    particles->acc_col = (double*) allocate_aligned(particles->N * sizeof(double));
    if (particles->acc_col == NULL) {
        printf("Error while allocating memory for the particles forces x\n");
        destroy_particles(particles);
        return NULL;
    }

    particles->acc_row = (double*) allocate_aligned(particles->N * sizeof(double));
    if (particles->acc_row == NULL) {
        printf("Error while allocating memory for the particles forces y\n");
        destroy_particles(particles);
        return NULL;
    }

    memset(particles->vel_col, 0, particles->N * sizeof(double));
    memset(particles->vel_row, 0, particles->N * sizeof(double));
    memset(particles->acc_col, 0, particles->N * sizeof(double));
    memset(particles->acc_row, 0, particles->N * sizeof(double));

    // auto vectorize 16 byte vecs
    particles->max_acc_col = 0.0;
    particles->max_acc_row = 0.0;

    // Init particles
    place_particles(particles, params->grid.BoxSize, params->system.A_deltaPar);

    // Initialization successful
    return particles;
}

// G' normalized on parameters
double compute_Gprime(NormalizationParams* norm) {
    return G_SI * norm->UnitMass * (norm->UnitTime * norm->UnitTime) / (norm->UnitLength * norm->UnitLength * norm->UnitLength);
}

void place_particles(Particles* particles, vec2d_t BoxSize, double A_deltaPar) {
    srand48(42); // Fixed seed for reproducibility

    int N = particles->N;
    // If N is not a perfect square, still create a near-square lattice and fill row-major.
    int ncols = (int)ceil(sqrt((double)N));
    int nrows = (int)ceil((double)N / (double)ncols);
    double dx_grid = BoxSize[_col_] / (double)ncols;
    double dy_grid = BoxSize[_row_] / (double)nrows;

    double k_col = 2 * M_PI / BoxSize[_col_];
    double k_row = 2 * M_PI / BoxSize[_row_];
    double half_pi = M_PI/2;

    // Simple 2D extension of 1D Zel'dovich scaling: displacement amplitude ~ A_deltaPar / k.
    double psi_amp_col = (fabs(k_col) > 0.0) ? (A_deltaPar / k_col) : 0.0;
    double psi_amp_row = (fabs(k_row) > 0.0) ? (A_deltaPar / k_row) : 0.0;

    #if defined(OMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (uint i = 0; i < N; i++) {
        double q_col, q_row;

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

        double psi_col = psi_amp_col * cos(k_col * q_col - half_pi);
        double psi_row = psi_amp_row * cos(k_row * q_row - half_pi);

        particles->pos_col[i] = q_col + psi_col;
        particles->pos_row[i] = q_row + psi_row;
    }
}

Mesh* init_mesh(vec2_t Ngrid) {

    #if defined(OMP)
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
    #endif

    Mesh* mesh = (Mesh*) malloc(sizeof(Mesh));
    if (mesh == NULL) {
        printf("Error while allocating memory for the mesh\n");
        return NULL;
    }

    // FFTW r2c_2d(n0, n1, ...) output is sized: n0 * (n1/2 + 1)
    // Here: n0 = Ngrid[_row_], n1 = Ngrid[_col_]
    size_t size = (size_t) Ngrid[_row_] * Ngrid[_col_];
    size_t fft_size = (size_t) Ngrid[_row_] * (Ngrid[_col_] / 2.0 + 1);
    mesh->grid_size = size;

    // allocate memory for the FFTW arrays
    mesh->kDensity = (fftw_complex*) fftw_alloc_complex(fft_size);
    mesh->kPot = (fftw_complex*) fftw_alloc_complex(fft_size);

    // allocate memory for the real-space arrays
    mesh->density = (double*) allocate_aligned(size * sizeof(double));
    mesh->pot = (double*) allocate_aligned(size * sizeof(double));

    // allocate memory for the forces arrays
    mesh->forces_x = (double*) allocate_aligned(size * sizeof(double));
    mesh->forces_y = (double*) allocate_aligned(size * sizeof(double));

    if (!mesh->kDensity || !mesh->kPot || !mesh->density || !mesh->pot) {
        printf("Error while allocating mesh arrays\n");
        destroy_mesh(mesh);
        return NULL;
    }

    // zero-initialize real-space arrays for deterministic transforms
    memset(mesh->density, 0, size * sizeof(double));
    memset(mesh->pot, 0, size * sizeof(double));

    // back and forth FFT plans (note: FFTW takes int dimensions)
    mesh->fft_real_fwd = fftw_plan_dft_r2c_2d((int) Ngrid[_row_], (int) Ngrid[_col_], mesh->density, mesh->kDensity, FFTW_MEASURE);
    mesh->fft_real_bck = fftw_plan_dft_c2r_2d((int) Ngrid[_row_], (int) Ngrid[_col_], mesh->kPot, mesh->pot, FFTW_MEASURE);
    if (!mesh->fft_real_fwd || !mesh->fft_real_bck) {
        printf("Error while creating FFTW plans\n");
        destroy_mesh(mesh);
        return NULL;
    }

    return mesh;
}

int destroy_mesh(Mesh* mesh) {
    #ifdef OMP
    fftw_cleanup_threads();
    #endif
    if (mesh->fft_real_fwd) fftw_destroy_plan(mesh->fft_real_fwd);
    if (mesh->fft_real_bck) fftw_destroy_plan(mesh->fft_real_bck);
    if (mesh->kDensity) fftw_free(mesh->kDensity);
    if (mesh->kPot) fftw_free(mesh->kPot);
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