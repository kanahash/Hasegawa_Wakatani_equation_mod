#include "MHW.h"

int main(int argc, char *argv[]) {
    // --- 修正点: 未使用パラメータの抑制 ---
    (void)argc;
    (void)argv;
    
    // Simulation parameters
    const int nx = 256;
    const int ny = 256;
    const int nt = 200000;
    const int isav = 25;
    const double kap = 1.0;
    const double alph = 1.0;
    const double nu = 0.22;
    const double mu = 1e-4;
    const double dt = 1e-2;
    const double lx = 10.0 * M_PI;
    const double ly = 32.0 * M_PI;
    const int shot_no = 13;
    char dir[128];
    sprintf(dir, "data%d/", shot_no); // Directory path

    // Grid setup
    double dx = lx / nx;
    double dy = ly / ny;
    double *x = (double *)malloc(nx * sizeof(double));
    double *y = (double *)malloc(ny * sizeof(double));
    for (int i = 0; i < nx; i++) x[i] = i * dx;
    for (int j = 0; j < ny; j++) y[j] = j * dy;

    // Initial conditions
    double **n_init = alloc_2d_double(ny, nx);
    double **phi_init = alloc_2d_double(ny, nx);
    const double s = 2.0;
    const double s2 = s * s;
    const double lx_half = lx / 2.0;
    const double ly_half = ly / 2.0;
    
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double r1 = pow(x[i] - lx_half, 2) + pow(y[j] - ly_half, 2);
            n_init[j][i] = 0.1 * exp(-r1 / s2);
            phi_init[j][i] = n_init[j][i];
        }
    }

    // Run the simulation
    MHW(nx, ny, lx, ly, nt, dt, kap, alph, mu, nu, phi_init, n_init, isav, dir);

    // Cleanup
    free(x);
    free(y);
    free_2d_double(n_init);
    free_2d_double(phi_init);
    
    // Cleanup for FFTW (not strictly required here but good practice)
    fftw_cleanup(); 

    return 0;
}
