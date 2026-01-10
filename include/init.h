#ifndef INIT_H
#define INIT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "global.h"

// function to read the parameters from a file
Params* read_params(char* filename);

double compute_Gprime(NormalizationParams* norm);

Particles* init_particles(Params* params);
void place_particles(Particles* particles, vec2d_t BoxSize, double A_deltaPar);
int destroy_particles(Particles* particles);

Mesh* init_mesh(vec2_t Ngrid);
int destroy_mesh(Mesh* mesh);


#endif // INIT_H