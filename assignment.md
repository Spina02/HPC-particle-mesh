# Particle-Mesh

## Problem Definition

### Vlasov-Poisson equations

We want to integrate equations for the evolutions of a colisionless fluid acting under the action of gravity. The equations governing the system are the Vlasov-Poisson:

$$
\dfrac{\partial f}{\partial t} + v \cdot \dfrac{\partial f}{\partial x} - \nabla V \cdot \dfrac{\partial f}{\partial v} = 0
$$

where the distribution function $f(x,v,t)$ depends on _position_, _velocities_ and _time_.

The density is: 

$$
\rho(x,t) = \int f(x, v, t) dv
$$


Such equations (in 3D) lives in a 6D+1 dimensions, and their direct integration (using e.g. Boltzmann codes) is highly inefficient.

Thus, we can obtain the momentum equation by multiplying VP equations by $v$ and integrating over velocities:

$$
\dfrac {\partial \langle v \rangle}{\partial t} + \langle v \rangle \cdot \nabla \langle v \rangle = - \nabla V - \dfrac 1\rho \nabla \cdot \mathbb P_J
$$

where the _Pressure tensor_ is:

$$
\mathbb P_J \equiv \rho \sigma^2_{ij} = \rho( \langle v_iv_j \rangle - \langle v_i \rangle \langle v_j \rangle)
$$

These are simply the equations of a compressible fluid supported by pressure in the form of velocity dispersion. Choosing a particular solution, called single speed because the velocities are a function of the positions (no multiple velocities at the same point in space):

$$
f(x, v, t) = \rho(x, t) \delta(v - u (x, t))
$$

VP equations become the Euler-Poisson equations:

$$
\dfrac{\partial \rho}{\partial t} + \nabla \cdot (\rho u) = 0
\\[1em]
\dfrac {\partial u}{\partial t} + (u \cdot \nabla) u = - \nabla V
\\[1em]
\nabla^2 V = 4 \pi G (\rho - \bar \rho)
$$

These are familiar, but becomes undefined at shell crossing (where more fluid elements do cross) because the density becomes infinite.

Using instead an isotropy _ansatz_ for the pressure tensor:
$$
\mathbb P_J = P(\rho) \mathbb I
$$

we simply introduce an equation of state of the kind 
$$
P = P(\rho)
$$

and obtain equations similar to those used for gases:
$$
\dfrac {\partial u}{\partial t} + u \cdot \nabla u = - \nabla V - \dfrac 1 \rho \nabla P
$$

but here $P$ is not the hydrodinamical pressure.

If we sample the fluid with massive particles (N-Body codes), evolve them under the action of gravity and allow them to cross, we obtain a system that is Vlasov_Poisson at all scales above the scale that define the massive particle themselves.

However, doing so we do not exacly resolve the VP equations! We work in configuration space, and don't explicitly follow the evolution of the momentum part of the phase space.

There are several numerical codes to deal with this problem, most diffuse ones being direct codes, tree codes, particle-mesh codes and combinations of them.

## Particle Mesh scheme

A Particle-Mesh code uses massive particle to define a density field on a mesh, an solves the Poisson equation such a mesh. Then, it evaluates the forces on the mesh, interpolates them to particles, and drift particles. This scheme is particularly fast because it solves the Poisson equation in the Fourier space, making use of a numerical algorithm called Fast Fourier Transform that is very efficient.

The Particle-Mesh scheme is thus:

- the density field is estimated on a grid using massive particles, representing fluid elements.
- the density field is transformed to the Fourier space
- the gravitational potential is computed using the green function of the Laplacian
- the gravitational potential is transformed back and forces are evaluated
- forces (on the mesh) are interpolated to particle
- particles velocities are updated with a chosen time-step
- particles positions are then updated, usually with a leap-frog integrator
- cycle is repeated until the desired final time.

We will write a simple, non-cosmological unidimensional PM code.

<!-- I'll actually do a 2D particle mesh -->

### Initial conditions

We need an external code to generate initial conditions (ICs) of the form $x,v$ for $N$ particles. To do this, we need to fix the units of measurements (UoM).

Use a parameter file to set UdM and other relevant characteristics.ù

We will use commonly used Udm:

- `UnitVel` $1e5$ ($1km/s$)
- `UnitMass` $1.989e43$ ($10^{10} M_\odot$)
- `UnitLength` $3.085678e21$ ($1 kpc$)

...compute the unit of time and the value of G!

Other needed parameters will be:

- `N_points`: number of mass points

- `N_grid`: FFT grid

- `BoxLenght` Length of the box (in kpc)

- `A_deltaPar` Maximum density contrast

other useful values could also be added to the parameter file (e.g. `H0`, `rho_crit`...)

We will evolve a sinusoudal density contrast: 

$$
\delta = A \sin (x \cdot 2 \pi / L - \pi / 2)
$$

A will be small because the initial density contrast must be linear. Velocities will be set to zero.

Note that using physical UoM means that we have a 3D distribution with only radial (1D) perturbations. Clearly, in reality this setup would not be stable against perturbations in the other two spherical coordinates.

The IC code should be able to write output in binary or in text, on request. It should also produce an histogram of the density to verify its correctness.

### Density computation

Once we have the (initial or evolved) particle distribution, we must coompute the density on a grid, to be able to perform the Fourier transform and evaluate the potential.

The simplest way is to simply assign the mass of each particle to the grid cell it belongs. This scheme is called **Nearest Grid Point** (NGP).

$$
W_1(x) = 1, \qquad |x| < 1
$$

The problem is that the density field, using NGP, is discontinuos. One can then "smooth" the mass on the two nearest grid cells, proportionally to the distance of the particle to the grid cell center. This scheme is called **Cloud in Cell** (CIC) and gives a continuos density field but a discontinuos first derivative.

$$
W_2(x) = 1- |x|, \qquad |x| < 1
$$

The continuity can be assured for higher order derivatives using more cells. Assigning the mass on three cells is done with the **Triangular shaped cloud** (TSC) scheme.


$$
\begin{cases}
\frac 34 - x^2, & |x| < \frac 13 \\
\frac 12 \left( \frac 32 - |x| \right)^2, & \frac 12 < |x| < \frac 32
\end{cases}
$$

The density in each cell will be:

$$
\rho_j = \sum_{i=1}^{N_p} m_i W (x_i - x_j)
$$

where $\rho_j$ is the density in the grid cell, $x_i$ the particle position, $N_p$ the particle number and $x_j$ the position of the center of the grid cell. 

> Note that $W(x)$ is zero if more than $1,2,3$ cells are considered.

Clearly, the higher the order of the assignment scheme is, the more precise is the computation.

The function that evaluate densities should be able to use one of the above scheme, depending on one appropriate `#define`. For testing purpose, print out the produced densities and plot them.

### Fast Fourier Transform and potential computation

The numerical solution of the Poisson equation would require a double integral, that is not very stable and is unefficient. By transforming the density in the Fourier space, the Laplacian operator becomes a simple multiplication: 

$$
V_k = G_k \delta_k = -1/k^2\delta_k
$$

Actually $G_k = - 1/k^2$ is called _poor man green function_! This is because it is the Green function of the Laplace operator only for infinite continous domains. Keeping in consideration a finite box and the discretization of the space, 

$$
G_k = - \dfrac{(\Delta x / 2)^2}{[\sin^2(k_x/2)]}
$$

where $\Delta_x$ is the grid spacing and $k_x = 2\pi / \Delta x_i$ (with $i$ ranging from 0 to $n_{grid}$) the wave numbers. ...but we will use in a first approach the poor man one.

To implement the potential computation we will use the FFTW 3.x library. The basics are:

- **definition and allocation of FFT data**:

    ```C
    #include "fftw3.h"

    fftw_complex *kDensity, *kPot;

    fftw_plan fft_real_fwd, fft_real_bck;

    kDensity = fftw_alloc_complex( Ngrid );

    kPot = fftw_alloc_complex( Ngrid );

    Pot = (double*) malloc( Ngrid * sizeof(double) );

    Density = (double*) malloc( Ngrid * sizeof(double) );
    ```

- **definition of back and fourth FFT**:

    ```C
    fft_real_fwd = fftw_plan_dft_r2c_2d(Ngrid, Density, kDensity, FFTW_ESTIMATE );

    fft_real_bck = fftw_plan_dft_c2r_2d(Ngrid, kPot, Pot, FFTW_ESTIMATE );
    ```

- **execution of direct FFT**:

    ```C
    fftw_execute( fft_real_fwd );
    ```

- **computation of the potential in the Fourier space**:

    ```C
    norm = 2 * M_PI / BoxSize;

    for( int i=1; i<Ngrid/2+1; i++) {
        k = (i*1.0) * norm;
        kPot[i][0] = -kDensity[i][0]/k/k;
        kPot[i][1] = -kDensity[i][1]/k/k;
    }
    ```

    > Note that the Fourier transform of the density is complex...

    > Note the definition of the wave numbers.

- **execution of the inverse FFT and normalization**:
    ```C
    fftw_execute( fft_real_bck );

    /* normalize */
    double norm = 1.0 / Ngrid;

    for( int i=0; i<Ngrid; i++)
        Pot[i] *= norm;
    ```

    ...at this point you have the gravitational potential, ready to be differentiated to get the force.

When implementing this part, make tests to be sure that the machinery is working. This can be easily done since in 1D, the Poisson equation has an analitic solution for a number of forms for $\rho - \bar \rho$!

### Force computation and interpolation

Once we obtain the potential, the force acting on each particle is simply $F = -\nabla V$. This can be obtained on the grid by simple discrete differentiation:

$$
F_i = - \dfrac{V_{i+1} - V_{i-1}}{2 \Delta x}
$$

It is very important to use the same scheme used in the density evaluation to obtain the forces on the particles, thus:

$$
F_p = \sum_{i = 1}^{N_{grid}} W (x_p - x_i) F_i
$$

to avoid "self-forces" (forces produced by the particle on itself)

### Leap Frog

At this point we have the acceleration acting on each particle and we can update velocities and positions:

$$
v_i^{t+1/2 \Delta t} = v_i^{t-1/2\Delta t} + (F_i / m_i) * \Delta t
\\
x_i^{t + \Delta t} = x_i^t + v_i^{t+1/2\Delta t} \Delta t
$$

Note that we would initially need $v_i^{t-1/2\Delta t}$; starting with null initial velocities, a reasonable approximation is to alto put this to null.

It is important to observe that our simulations will be periodic given that the potential solver is periodic. Thus, periodic boundary conditions must be implemented.

The simulation will continue from $t = 0$ to $t = t_{final}$.

$t_{final}$ must be given in the parameter file, and initial time step must also be guven by hand, a good choice being $t_{final} / N_{timesteps}$ where $N_{timesteps}$ is an estimate of the total number of timesteps needed to complete the run. This must be found empirically but a good estimate is 1000-10000.

The timestep must be adapted during the simulation. Ideally, a particle should not move more than a fraction of the grid size in a timestep.

A simple choice would thus be $\Delta t = \omega \Delta x / \min (v_i)$, with $\omega \approx 1/2$, under the hypothesis that the acceleration does not have large changes between 2 timesteps. Note that in a PM code the timestep is unique for all particles.

Gadget uses

$$
\Delta t = \sqrt {\dfrac{2 \eta \epsilon}{|a|}}
$$

where $\epsilon$ is the force softening, $|a|$ the modulus of acceleration and $\eta$ a parameter set to $0.05 - 0.1$

We can simply adopt the first option.

More details on the writing of a (3D) PM code can be found in Hockney & Eastwood, "Computer simulations using particles", 1981.

---

Code written in several files

- one to initialize the code
- one for density computation
- one for force computation
- main.c -> calls the functions to implement the main pipeline

common function to read the parameter file

use makefile to compile



---

checkpointing?