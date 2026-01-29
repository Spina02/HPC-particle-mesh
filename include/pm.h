#ifndef PM_H
#define PM_H

#include "global.h"
#include <sys/param.h>
#include <stdbool.h>
#include "rsort.h"

#define METHOD "TSC"

#ifdef __NVCOMPILER
#define PRAGMA_UNROLL _Pragma("unroll 3")
#else
#define PRAGMA_UNROLL _Pragma("GCC unroll 3")
#endif

// density estimation function
real_t* estimate_density(Mesh* restrict mesh, GridParams* restrict grid, Particles* restrict particles);

// potential computation functions
void compute_potential(Mesh* restrict mesh, vec2_t Ngrid, vec2d_t BoxSize);

// force computation functions
void compute_forces(Mesh* restrict mesh, vec2d_t BoxSize, vec2_t Ngrid);

// force interpolation functions
void interpolate_forces(Mesh* restrict mesh, Particles* restrict particles, vec2d_t BoxSize, vec2_t Ngrid);

// particle update functions
void drift_particles(Particles* restrict particles, real_t dt, vec2d_t BoxSize);
void kick_particles(Particles* restrict particles, real_t dt);

// helper inline functions

#ifdef USE_GPU
#pragma acc routine seq
#endif
inline static real_t TSC_weight(real_t d) {
    #if defined(VEC) || defined(USE_OMP) || defined(USE_GPU)
        real_t w1 = (real_t)(0.75 - d*d);
        real_t w2 = ((real_t)0.5*((real_t)1.5-d)*((real_t)1.5-d));
        return (d<(real_t)0.5)*w1+(d>=(real_t)0.5 && d<(real_t)1.5)*w2;
    #else
        if (d < (real_t)0.5) // 3/4 - |x|^2
            return 0.75 - d * d;
        else if (d>(real_t)0.5 && d<(real_t)1.5)        // 1/2 (3/2 - |x|)^2
            return (real_t)0.5*((real_t)1.5-d)*((real_t)1.5-d);
        return (real_t)0.0;
    #endif
}


#ifdef POW2GRID
#ifdef USE_GPU
#pragma acc routine seq
#endif
inline static int fast_mod(int grid_idx, uint size) {
    return grid_idx & (size - 1);
}
#else
#ifdef USE_GPU
#pragma acc routine seq
#endif
inline static int fast_mod(int grid_idx, uint size) {
    bool out_low = grid_idx < 0.0;
    bool out_high = grid_idx >= size;
    return (int) (grid_idx + out_low * size - out_high * size);
}
#endif

#ifdef USE_GPU
#pragma acc routine seq
#endif
inline static real_t fast_fmod(real_t grid_idx, uint size) {
    bool out_low = grid_idx < 0.0;
    bool out_high = grid_idx >= size;
    return grid_idx + out_low * size - out_high * size;
}

void reorder_particles(Particles* restrict p, GridParams* restrict grid, real_t* restrict tmp_arrays);

#endif // PM_H
