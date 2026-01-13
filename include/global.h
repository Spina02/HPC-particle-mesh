#ifndef GLOBAL_H
#define GLOBAL_H

#include <fftw3.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

#define G_SI 6.67e-11
extern double G_prime;

typedef unsigned int uint;

typedef uint vec2_t[2];
typedef double vec2d_t[2];

typedef struct Particles {
    
    double* restrict pos_col;
    double* restrict pos_row;

    double* restrict vel_col;
    double* restrict vel_row;
    
    double* restrict acc_col;
    double* restrict acc_row;
    
    double mass;

    double max_acc_col;
    double max_acc_row;

    uint N;
} Particles;

// struct to handle normalization parameters

typedef struct NormalizationParams {
    double UnitVel;
    double UnitMass;
    double UnitLength;
    double UnitTime;
} NormalizationParams;

// struct to handle grid parameters

typedef struct GridParams {
    uint Npoints;
    vec2_t Ngrid;
    vec2d_t BoxSize;
} GridParams;

typedef struct SystemParams {
    double A_deltaPar;
    uint n_iter;
} SystemParams;

typedef struct Params {
    NormalizationParams norm;
    GridParams grid;
    SystemParams system;
} Params;

typedef struct Mesh {
    fftw_complex* restrict kDensity;
    fftw_complex* restrict kPot;
    double* restrict density;
    double* restrict pot;
    double* restrict forces_x;
    double* restrict forces_y;
    fftw_plan fft_real_fwd;
    fftw_plan fft_real_bck;
    size_t grid_size;
} Mesh;

#endif // GLOBAL_H