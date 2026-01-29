#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef USE_OMP
    #include <omp.h>
#endif

#ifdef __cplusplus
/* Make C99 'restrict' compatible with C++ compilers (e.g., nvc++, nvcc host) */
#ifndef restrict
#define restrict __restrict__
#endif
#endif

// ------------------------------------------
// 1. GENERAL SETTINGS
// ------------------------------------------

// debug print macro
#ifdef DEBUG
    #define debug_print(...)        printf(__VA_ARGS__)
#else
    #define debug_print(...)
#endif

#define _col_ 0
#define _row_ 1

#ifdef NRANDOM
    #define RANDOM 0
#else
    #define RANDOM 1
#endif

#define ALIGNMENT 64

#define THREADS_LIMIT 32

// ------------------------------------------
// 2. PRECISION AND MATH MACROS
// ------------------------------------------

#ifdef USE_FLOAT // single precision
    typedef float real_t;
    #define REAL_FMT "%f"
    #define M_PI_T      3.14159265358979323846f
    #define G_SI        6.67e-11f
    // name mangling
    #define M_F(func)    func ## f      // sqrt -> sqrtf
    #define FFTW_N(name) fftwf_ ## name // execute -> fftwf_execute

#else // double precision
    typedef double real_t;
    #define REAL_FMT "%lf"
    #define M_PI_T      3.14159265358979323846
    #define G_SI        6.67e-11
    // name mangling
    #define M_F(func)    func           // sqrt -> sqrt
    #define FFTW_N(name) fftw_ ## name  // execute -> fftw_execute
#endif

// Unified Math Macros
#define SQRT(x)     M_F(sqrt)(x)
#define SIN(x)      M_F(sin)(x)
#define COS(x)      M_F(cos)(x)
#define FABS(x)     M_F(fabs)(x)
#define CEIL(x)     M_F(ceil)(x)
#define MAX(x,y)    ((x) > (y) ? (x) : (y))
#define MIN(x,y)    ((x) < (y) ? (x) : (y))

typedef unsigned int uint;
typedef uint   vec2_t[2];   // Integer vector
typedef real_t vec2d_t[2];  // Real vector
extern real_t G_prime;      // Declared here, defined in main.c

// ------------------------------------------
// 3. FFT ABSTRACTION LAYER
// ------------------------------------------

#ifdef USE_GPU // GPU
    #include <cufft.h>
    #include <openacc.h>

    #define FFTW_INIT_THREADS
    #define FFTW_PLAN_WITH_NTHREADS
    #define FFTW_CLEANUP_THREADS

    // GPU types
    #ifdef USE_FLOAT
        typedef cufftComplex complex_t;
        typedef cufftReal    cufft_real_t;
        #define CUFFT_TYPE_R2C     CUFFT_R2C
        #define CUFFT_TYPE_C2R     CUFFT_C2R
        #define CUFFT_R2C_FUNC     cufftExecR2C
        #define CUFFT_C2R_FUNC     cufftExecC2R
    #else
        typedef cufftDoubleComplex complex_t;
        typedef cufftDoubleReal    cufft_real_t;
        #define CUFFT_TYPE_R2C     CUFFT_D2Z
        #define CUFFT_TYPE_C2R     CUFFT_Z2D
        #define CUFFT_R2C_FUNC     cufftExecD2Z
        #define CUFFT_C2R_FUNC     cufftExecZ2D
    #endif

    typedef cufftHandle pm_plan_t;

    #define C_RE(c) (c).x
    #define C_IM(c) (c).y

    // GPU memory management (host pinned/aligned memory)
    static inline void* pm_malloc(size_t n, size_t size) { 
        void* ptr = NULL; 
        if (posix_memalign(&ptr, ALIGNMENT, n * size) != 0) return NULL; 
        return ptr; 
    }
    static inline void pm_free(void* ptr) { free(ptr); }

    // GPU FFTW Function Wrappers
    #define PM_FFTW_FWD             CUFFT_R2C_FUNC
    #define PM_FFTW_BCK             CUFFT_C2R_FUNC
    #define PM_DESTROY_PLAN         cufftDestroy

#else // CPU
    #include <fftw3.h>

    // CPU types
    typedef FFTW_N(complex) complex_t;
    typedef FFTW_N(plan)    pm_plan_t;

    // CPU FFTW Thread Management
    #ifdef USE_OMP
        #define FFTW_INIT_THREADS       FFTW_N(init_threads)()
        #define FFTW_PLAN_WITH_NTHREADS FFTW_N(plan_with_nthreads)(omp_get_max_threads())
        #define FFTW_CLEANUP_THREADS    FFTW_N(cleanup_threads)()
    #else
        #define FFTW_INIT_THREADS
        #define FFTW_PLAN_WITH_NTHREADS
        #define FFTW_CLEANUP_THREADS
    #endif
    
    #define FFTW_PLAN_DFT_R2C_2D    FFTW_N(plan_dft_r2c_2d)
    #define FFTW_PLAN_DFT_C2R_2D    FFTW_N(plan_dft_c2r_2d)

    // CPU Complex accessor macros
    #define C_RE(c) ((c)[0])
    #define C_IM(c) ((c)[1])

    // CPU Memory Management (FFTW Optimized)
    static inline void* pm_malloc(size_t n, size_t size) { (void)size; return FFTW_N(alloc_complex)(n); }
    static inline void pm_free(void* ptr) { FFTW_N(free)(ptr); }
    
    // CPU FFTW Function Wrappers
    #define PM_FFTW_FWD             FFTW_N(execute)
    #define PM_FFTW_BCK             FFTW_N(execute)
    #define PM_DESTROY_PLAN         FFTW_N(destroy_plan)
#endif

// ------------------------------------------
// 4. STRUCTURES DEFINITIONS
// ------------------------------------------

typedef struct Particles {
    
    real_t* restrict pos_col;
    real_t* restrict pos_row;

    real_t* restrict vel_col;
    real_t* restrict vel_row;
    
    real_t* restrict acc_col;
    real_t* restrict acc_row;
    
    real_t mass;

    real_t max_acc_col;
    real_t max_acc_row;

    uint N;
} Particles;

// struct to handle normalization parameters

typedef struct NormalizationParams {
    real_t UnitVel;
    real_t UnitMass;
    real_t UnitLength;
    real_t UnitTime;
} NormalizationParams;

// struct to handle grid parameters

typedef struct GridParams {
    uint Npoints;
    vec2_t Ngrid;
    vec2d_t BoxSize;
} GridParams;

typedef struct SystemParams {
    real_t A_deltaPar;
    uint n_iter;
} SystemParams;

typedef struct Params {
    NormalizationParams norm;
    GridParams grid;
    SystemParams system;
} Params;

typedef struct Mesh {
    complex_t* restrict kDensity;
    complex_t* restrict kPot;
    pm_plan_t plan_fwd;
    pm_plan_t plan_bck;
    real_t* restrict density;
    real_t* restrict pot;
    real_t* restrict forces_x;
    real_t* restrict forces_y;
    size_t grid_size;
} Mesh;

typedef struct {
    uint particle_index;
    uint cell_index;
} SortItem;

#endif // GLOBAL_H
