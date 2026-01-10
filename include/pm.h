#ifndef PM_H
#define PM_H

#include "global.h"
#include <sys/param.h>

// define the function to estimate the density via macro
#ifdef NGP
    #define METHOD "NGP"
    #define estimate_density(...) estimate_density_NGP(__VA_ARGS__)
    #define interpolate_forces(...) interpolate_forces_NGP(__VA_ARGS__)
#elif defined(CIC)
    #define METHOD "CIC"
    #define estimate_density(...) estimate_density_CIC(__VA_ARGS__)
    #define interpolate_forces(...) interpolate_forces_CIC(__VA_ARGS__)
#else // #elif defined(TSC)
    #define METHOD "TSC"
    #define estimate_density(...) estimate_density_TSC(__VA_ARGS__) // default
    #define interpolate_forces(...) interpolate_forces_TSC(__VA_ARGS__)
#endif

// density estimation functions and weight functions
double* estimate_density_NGP(Mesh* mesh, GridParams* grid, Particles* particles);
double* estimate_density_CIC(Mesh* mesh, GridParams* grid, Particles* particles);
double* estimate_density_TSC(Mesh* mesh, GridParams* grid, Particles* particles);

double NGP_weight(double dist);
double CIC_weight(double dist);
double TSC_weight(double dist);

// potential computation functions
void compute_potential(Mesh* mesh, vec2_t Ngrid, vec2d_t BoxSize);

// force computation functions
void compute_forces(Mesh* mesh, vec2d_t BoxSize, vec2_t Ngrid);

// force interpolation functions
void interpolate_forces_NGP(Mesh* mesh, Particles* particles, vec2d_t BoxSize, vec2_t Ngrid);
void interpolate_forces_CIC(Mesh* mesh, Particles* particles, vec2d_t BoxSize, vec2_t Ngrid);
void interpolate_forces_TSC(Mesh* mesh, Particles* particles, vec2d_t BoxSize, vec2_t Ngrid);

// particle update functions
void drift_particles(Particles* particles, double dt, vec2d_t BoxSize);
void kick_particles(Particles* particles, double dt);

#endif // PM_H