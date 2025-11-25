#include "MHW.h"

// Main simulation function
void MHW(int nx, int ny, double lx, double ly, int nt, double dt, double kap, double alph, double mu, double nu, double **phi_init, double **n_init, int isav, const char *dir) {
    NX = nx; NY = ny; LX = lx; LY = ly;
    double dx = lx / nx;
    double dy = ly / ny;
    int nsav = nt / isav;

    // 1. Setup wave number and grid arrays (Global: KX, KY, KX2, KY2, KXD, KYD)
    KX_real = (double *)malloc(nx * sizeof(double));
    KY_real = (double *)malloc(ny * sizeof(double));
    KX = alloc_2d_cplx(ny, nx); KY = alloc_2d_cplx(ny, nx);
    KX2 = alloc_2d_cplx(ny, nx); KY2 = alloc_2d_cplx(ny, nx);
    KXD = alloc_2d_cplx(ny, nx); KYD = alloc_2d_cplx(ny, nx);

    for (int i = 0; i < nx; i++) {
        KX_real[i] = 2.0 * M_PI / lx * (i < nx / 2 ? (double)i : (double)i - nx);
    }
    for (int j = 0; j < ny; j++) {
        KY_real[j] = 2.0 * M_PI / ly * (j < ny / 2 ? (double)j : (double)j - ny);
    }

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            KX[j][i] = KX_real[i];
            KY[j][i] = KY_real[j];
            KX2[j][i] = KX_real[i] * KX_real[i];
            KY2[j][i] = KY_real[j] * KY_real[j];

            // De-aliasing filter (1 for first/last third, 0 otherwise)
            KXD[j][i] = (i < nx / 3 || i >= 2 * nx / 3) ? 1.0 : 0.0;
            KYD[j][i] = (j < ny / 3 || j >= 2 * ny / 3) ? 1.0 : 0.0;
        }
    }

    // 2. Initial state FFT
    fftw_plan p_fft_phi, p_fft_n, p_ifft_zeta;
    cplx **phif = alloc_2d_cplx(ny, nx);
    cplx **nf = alloc_2d_cplx(ny, nx);
    cplx **zetaf = alloc_2d_cplx(ny, nx);

    p_fft_phi = fftw_plan_dft_2d(ny, nx, (fftw_complex *)phi_init[0], phif[0], FFTW_FORWARD, FFTW_ESTIMATE);
    p_fft_n = fftw_plan_dft_2d(ny, nx, (fftw_complex *)n_init[0], nf[0], FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p_fft_phi);
    fftw_execute(p_fft_n);
    fftw_destroy_plan(p_fft_phi);
    fftw_destroy_plan(p_fft_n);

    // Initial zetaf = -k^2 * phif
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            zetaf[j][i] = -(KX2[j][i] + KY2[j][i]) * phif[j][i];
        }
    }

    // 3. Setup history arrays (t, y, x)
    // The Python code saves real space values: phi, n, zeta
    size_t hist_size = nsav * nx * ny;
    double *phihst_data = (double *)calloc(hist_size, sizeof(double));
    double *nhst_data = (double *)calloc(hist_size, sizeof(double));
    double *zetahst_data = (double *)calloc(hist_size, sizeof(double));

    // Save initial state (t=0)
    p_ifft_zeta = fftw_plan_dft_2d(ny, nx, zetaf[0], (fftw_complex *)zetahst_data, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_ifft_zeta);
    fftw_destroy_plan(p_ifft_zeta);

    // The other initial states are already real:
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            phihst_data[0 * ny * nx + j * nx + i] = phi_init[j][i];
            nhst_data[0 * ny * nx + j * nx + i] = n_init[j][i];
            // IFFT of zetaf needs normalization (and real part extraction for zeta)
            zetahst_data[j * nx + i] = creal(zetahst_data[j * nx + i]) / (NX * NY);
        }
    }
    
    // 4. Time stepping loop (4th-order Runge-Kutta with integrating factor)
    cplx **gw1 = alloc_2d_cplx(ny, nx), **ga1 = alloc_2d_cplx(ny, nx);
    cplx **gw2 = alloc_2d_cplx(ny, nx), **ga2 = alloc_2d_cplx(ny, nx);
    cplx **gw3 = alloc_2d_cplx(ny, nx), **ga3 = alloc_2d_cplx(ny, nx);
    cplx **gw4 = alloc_2d_cplx(ny, nx), **ga4 = alloc_2d_cplx(ny, nx);
    
    cplx **zetaf_temp = alloc_2d_cplx(ny, nx);
    cplx **nf_temp = alloc_2d_cplx(ny, nx);
    
    // Constant for integrating factor: exp(-mu * k^4 * dt)
    cplx **exp_factor = alloc_2d_cplx(ny, nx);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double k2 = creal(KX2[j][i]) + creal(KY2[j][i]);
            exp_factor[j][i] = cexp(-mu * k2 * k2 * dt);
        }
    }

    printf("Starting MHW simulation for %d timesteps...\n", nt);

    for (int it = 1; it < nt; it++) {
        // --- 4a. Integrating factor step ---
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf[j][i] *= exp_factor[j][i];
                nf[j][i] *= exp_factor[j][i];
            }
        }
        
        // --- 4b. 4th-order Runge-Kutta step ---
        
        // k1 = adv(zetaf, nf)
        adv(zetaf, nf, dx, dy, alph, nu, kap, gw1, ga1);

        // k2 = adv(zetaf + 0.5*dt*k1_zeta, nf + 0.5*dt*k1_n)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + 0.5 * dt * gw1[j][i];
                nf_temp[j][i] = nf[j][i] + 0.5 * dt * ga1[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw2, ga2);

        // k3 = adv(zetaf + 0.5*dt*k2_zeta, nf + 0.5*dt*k2_n)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + 0.5 * dt * gw2[j][i];
                nf_temp[j][i] = nf[j][i] + 0.5 * dt * ga2[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw3, ga3);

        // k4 = adv(zetaf + dt*k3_zeta, nf + dt*k3_n)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + dt * gw3[j][i];
                nf_temp[j][i] = nf[j][i] + dt * ga3[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw4, ga4);

        // Final update: u^{n+1} = u^n + dt/6 * (k1 + 2k2 + 2k3 + k4)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf[j][i] += dt / 6.0 * (gw1[j][i] + 2.0 * gw2[j][i] + 2.0 * gw3[j][i] + gw4[j][i]);
                nf[j][i] += dt / 6.0 * (ga1[j][i] + 2.0 * ga2[j][i] + 2.0 * ga3[j][i] + ga4[j][i]);
            }
        }
        
        // --- 4c. Save history ---
        if (it % isav == 0) {
            int t_idx = it / isav;
            fftw_plan p_ifft_current;

            // Calculate phif and enforce zero average
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    double k2 = creal(KX2[j][i]) + creal(KY2[j][i]);
                    phif[j][i] = k2 != 0.0 ? zetaf[j][i] / (-k2) : 0.0;
                }
            }
            phif[0][0] = 0.0;
            
            // IFFT for phi, n, zeta
            p_ifft_current = fftw_plan_dft_2d(ny, nx, phif[0], (fftw_complex *)&phihst_data[t_idx * ny * nx], FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(p_ifft_current);
            fftw_destroy_plan(p_ifft_current);

            p_ifft_current = fftw_plan_dft_2d(ny, nx, nf[0], (fftw_complex *)&nhst_data[t_idx * ny * nx], FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(p_ifft_current);
            fftw_destroy_plan(p_ifft_current);

            p_ifft_current = fftw_plan_dft_2d(ny, nx, zetaf[0], (fftw_complex *)&zetahst_data[t_idx * ny * nx], FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(p_ifft_current);
            fftw_destroy_plan(p_ifft_current);
            
            // Normalize and take real part
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    size_t idx = t_idx * ny * nx + j * nx + i;
                    // The history arrays were used as output buffers for FFTW, 
                    // so we need to access the real component and normalize.
                    phihst_data[idx] = creal(phihst_data[idx]) / (NX * NY);
                    nhst_data[idx] = creal(nhst_data[idx]) / (NX * NY);
                    zetahst_data[idx] = creal(zetahst_data[idx]) / (NX * NY);
                }
            }
            
            if (t_idx % (nsav/10) == 0) {
                 printf("Timestep: %d/%d (%.1f%%)\n", it, nt, (double)it / nt * 100.0);
            }
        }
    }
    
    // 5. Save results to binary files
    char path[256];
    
    // Create directory if it doesn't exist (equivalent to Python's os.makedirs)
    #if defined(_WIN32)
        _mkdir(dir);
    #else 
        mkdir(dir, 0777); // 0777 for permissions
    #endif

    FILE *fp;

    // Save phihst
    sprintf(path, "%sphi_shot%d.bin", dir, shot_no);
    fp = fopen(path, "wb");
    fwrite(phihst_data, sizeof(double), hist_size, fp);
    fclose(fp);
    printf("Saved phi history to %s\n", path);

    // Save nhst
    sprintf(path, "%snhst_shot%d.bin", dir, shot_no);
    fp = fopen(path, "wb");
    fwrite(nhst_data, sizeof(double), hist_size, fp);
    fclose(fp);
    printf("Saved n history to %s\n", path);

    // Save zetahst
    sprintf(path, "%szetahst_shot%d.bin", dir, shot_no);
    fp = fopen(path, "wb");
    fwrite(zetahst_data, sizeof(double), hist_size, fp);
    fclose(fp);
    printf("Saved zeta history to %s\n", path);

    // 6. Cleanup
    free(phihst_data); free(nhst_data); free(zetahst_data);
    
    free_2d_cplx(phif); free_2d_cplx(nf); free_2d_cplx(zetaf);
    
    free_2d_cplx(gw1); free_2d_cplx(ga1);
    free_2d_cplx(gw2); free_2d_cplx(ga2);
    free_2d_cplx(gw3); free_2d_cplx(ga3);
    free_2d_cplx(gw4); free_2d_cplx(ga4);
    free_2d_cplx(zetaf_temp); free_2d_cplx(nf_temp);
    free_2d_cplx(exp_factor);

    free(KX_real); free(KY_real);
    free_2d_cplx(KX); free_2d_cplx(KY); free_2d_cplx(KX2); free_2d_cplx(KY2); 
    free_2d_cplx(KXD); free_2d_cplx(KYD);
}
