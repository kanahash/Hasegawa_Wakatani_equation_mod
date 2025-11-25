#ifndef MHW_H
#define MHW_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// Define complex numbers
typedef double complex cplx;

// --- Global Variables Declaration (Defined in mhw_simulation.c) ---
extern int NX, NY;
extern double LX, LY;
extern int shot_no; 
extern cplx **KX, **KY, **KX2, **KY2, **KXD, **KYD;

// --- Memory Utilities Prototypes (memory_utils.c) ---
double **alloc_2d_double(int rows, int cols);
void free_2d_double(double **array);
cplx **alloc_2d_cplx(int rows, int cols);
void free_2d_cplx(cplx **array);

// --- MHW Simulation Prototypes (mhw_simulation.c) ---
void MHW(int nx, int ny, double lx, double ly, int nt, double dt, double kap, double alph, double mu, double nu, double **phi_init, double **n_init, int isav, const char *dir);

// --- Advection & Utility Prototypes (adv.c & adv_utils.c) ---
void adv(cplx **zetaf, cplx **nf, double dx, double dy, double alph, double nu, double kap, cplx **advff, cplx **advgf);

void calculate_phi_and_ifft(cplx **zetaf, cplx **nf, cplx **phif, double **phi, double **n, double **zeta);
void calculate_zonal_averages(double **phi, double **n, double dy, double LY, double *phiz, double *nz);

// [修正点: 特殊文字を排除し、整形]
void calculate_filtered_derivatives(cplx **phif, cplx **zetaf, cplx **nf, 
                                    cplx **phixf, cplx **phiyf, cplx **zetaxf, cplx **zetayf, cplx **nxf, cplx **nyf,
                                    double **phix, double **phiy, double **zetax, double **zetay, double **nnx, double **nny);

// [修正点: 特殊文字を排除し、整形]
void calculate_rhs_real_space(double **phi, double **n, double **zeta, double **phix, double **phiy, double **zetax, 
                              double **zetay, double **nnx, double **nny, double *phiz, double *nz,
                              double alph, double nu, double kap, double **advf, double **advg, cplx **phif);
                              
void calculate_fft_rhs(double **advf, double **advg, cplx **advff, cplx **advgf);


// --- MHW Utility Prototypes (MHW_utils.c) ---
void setup_grid_and_wavenumbers(int nx, int ny, double lx, double ly);

void initialize_state_and_history(int nx, int ny, int nsav, 
                                  double **phi_init, double **n_init, 
                                  cplx **phif, cplx **nf, cplx **zetaf, 
                                  double *phihst_data, double *nhst_data, double *zetahst_data);

void save_current_history(int t_idx, cplx **zetaf, cplx **nf, cplx **phif, 
                          double *phihst_data, double *nhst_data, double *zetahst_data);

// [修正点: 引数 exp_factor, phihst_data の型エラーを修正]
void save_and_cleanup(int nsav, const char *dir, 
                      cplx **phif, cplx **nf, cplx **zetaf, 
                      cplx **gw1, cplx **ga1, cplx **gw2, cplx **ga2, cplx **gw3, cplx **ga3, cplx **gw4, cplx **ga4, 
                      cplx **zetaf_temp, cplx **nf_temp, cplx *exp_factor, 
                      double *phihst_data, double *nhst_data, double *zetahst_data);

#endif // 
