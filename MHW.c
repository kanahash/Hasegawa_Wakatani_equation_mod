#include "MHW.h"

void MHW(int nx, int ny, double lx, double ly, int nt, double dt, double kap, double alph, double mu, double nu, double **phi_init, double **n_init, int isav, const char *dir) {
    // グローバル変数の設定
    NX = nx; NY = ny; LX = lx; LY = ly;
    double dx = lx / nx;
    double dy = ly / ny;
    int nsav = nt / isav;

    // Allocations
    cplx **phif = alloc_2d_cplx(ny, nx);
    cplx **nf = alloc_2d_cplx(ny, nx);
    cplx **zetaf = alloc_2d_cplx(ny, nx);
    
    // RK4 Buffers
    cplx **gw1 = alloc_2d_cplx(ny, nx), **ga1 = alloc_2d_cplx(ny, nx);
    cplx **gw2 = alloc_2d_cplx(ny, nx), **ga2 = alloc_2d_cplx(ny, nx);
    cplx **gw3 = alloc_2d_cplx(ny, nx), **ga3 = alloc_2d_cplx(ny, nx);
    cplx **gw4 = alloc_2d_cplx(ny, nx), **ga4 = alloc_2d_cplx(ny, nx);
    cplx **zetaf_temp = alloc_2d_cplx(ny, nx);
    cplx **nf_temp = alloc_2d_cplx(ny, nx);
    cplx **exp_factor = alloc_2d_cplx(ny, nx);
    
    // History Buffers
    size_t hist_size = (size_t)nsav * nx * ny;
    double *phihst_data = (double *)calloc(hist_size, sizeof(double));
    double *nhst_data = (double *)calloc(hist_size, sizeof(double));
    double *zetahst_data = (double *)calloc(hist_size, sizeof(double));

    // Setup & Initialize
    setup_grid_and_wavenumbers(nx, ny, lx, ly); 
    initialize_state_and_history(nx, ny, nsav, phi_init, n_init, phif, nf, zetaf, phihst_data, nhst_data, zetahst_data);
    
    // Integrating Factor
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double k2 = creal(KX2[j][i]) + creal(KY2[j][i]);
            exp_factor[j][i] = cexp(-mu * k2 * k2 * dt);
        }
    }

    printf("Starting MHW simulation...\n");

    // Time loop
    for (int it = 1; it < nt; it++) {
        // Step 1: Integrating factor
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf[j][i] *= exp_factor[j][i];
                nf[j][i] *= exp_factor[j][i];
            }
        }
        
        // Step 2: RK4
        adv(zetaf, nf, dx, dy, alph, nu, kap, gw1, ga1);

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + 0.5 * dt * gw1[j][i];
                nf_temp[j][i] = nf[j][i] + 0.5 * dt * ga1[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw2, ga2);

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + 0.5 * dt * gw2[j][i];
                nf_temp[j][i] = nf[j][i] + 0.5 * dt * ga2[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw3, ga3);

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + dt * gw3[j][i];
                nf_temp[j][i] = nf[j][i] + dt * ga3[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw4, ga4);

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf[j][i] += dt / 6.0 * (gw1[j][i] + 2.0 * gw2[j][i] + 2.0 * gw3[j][i] + gw4[j][i]);
                nf[j][i] += dt / 6.0 * (ga1[j][i] + 2.0 * ga2[j][i] + 2.0 * ga3[j][i] + ga4[j][i]);
            }
        }
        
        // Save History
        if (it % isav == 0) {
            int t_idx = it / isav;
            save_current_history(t_idx, zetaf, nf, phif, phihst_data, nhst_data, zetahst_data);
            if (t_idx % (nsav/10 + 1) == 0) printf("Timestep: %d/%d\n", it, nt);
        }
    }
    
    // Cleanup
    save_and_cleanup(nsav, dir, phif, nf, zetaf, gw1, ga1, gw2, ga2, gw3, ga3, gw4, ga4, zetaf_temp, nf_temp, exp_factor, phihst_data, nhst_data, zetahst_data);
}
