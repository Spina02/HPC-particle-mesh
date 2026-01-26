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
double* estimate_density(Mesh* restrict mesh, GridParams* restrict grid, Particles* restrict particles);

// potential computation functions
void compute_potential(Mesh* restrict mesh, vec2_t Ngrid, vec2d_t BoxSize);

// force computation functions
void compute_forces(Mesh* restrict mesh, vec2d_t BoxSize, vec2_t Ngrid);

// force interpolation functions
void interpolate_forces(Mesh* restrict mesh, Particles* restrict particles, vec2d_t BoxSize, vec2_t Ngrid);

// particle update functions
void drift_particles(Particles* restrict particles, double dt, vec2d_t BoxSize);
void kick_particles(Particles* restrict particles, double dt);

// helper inline functions

#ifdef USE_GPU
#pragma acc routine seq
#endif
inline static double TSC_weight(double dist) {
#if defined(VEC) || defined(GPU)
    double w1 = (0.75 - dist * dist);
    double w2 = 0.5 * (1.5 - dist) * (1.5 - dist);
    return (dist < 0.5) * w1 + (dist >= 0.5 && dist < 1.5) * w2;
#else
    // 3/4 - |x|^2
    if (dist < 0.5)
        return 0.75 - dist * dist;
    // 1/2 (3/2 - |x|)^2
    else if ((dist > 0.5) && (dist < 1.5))
        return 0.5 * (1.5 - dist) * (1.5 - dist);
    // else 0.0
    return 0.0;
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
inline static double fast_fmod(double grid_idx, uint size) {
    bool out_low = grid_idx < 0.0;
    bool out_high = grid_idx >= size;
    return grid_idx + out_low * size - out_high * size;
}

void reorder_particles(Particles* p, GridParams* grid, double* tmp_arrays);

#endif // PM_H
