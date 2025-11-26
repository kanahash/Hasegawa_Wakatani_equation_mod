#include "MHW.h"

void MHW(int nx, int ny, double lx, double ly, int nt, double dt, double kap, double alph, double mu, double nu, double **phi_init, double **n_init, int isav, const char *dir) {
    // グローバル変数の設定
    NX = nx; NY = ny; LX = lx; LY = ly;
    double dx = lx / nx;
    double dy = ly / ny;
    int nsav = nt / isav;
    if (nsav <= 0) nsav = 1;

    // --- 1. Allocations ---
    printf("Debug: Allocating memory...\n");
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
    
    if (!phif || !nf || !zetaf || !phihst_data) {
        fprintf(stderr, "Memory allocation failed in MHW.\n");
        return;
    }

    // --- 2. Setup & Initialize ---
    printf("Debug: Setting up grid...\n");
    setup_grid_and_wavenumbers(nx, ny, lx, ly); 
    
    printf("Debug: Initializing state...\n");
    initialize_state_and_history(nx, ny, nsav, phi_init, n_init, phif, nf, zetaf, phihst_data, nhst_data, zetahst_data);
    
    // 積分因子 exp_factor の計算
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double k2 = -(creal(KX2[j][i]) + creal(KY2[j][i]));
            exp_factor[j][i] = cexp(-mu * k2 * k2 * dt);
        }
    }

    printf("Starting MHW simulation for %d timesteps (saved steps: %d)...\n", nt, nsav);
    fflush(stdout); // 強制的にログを表示させる

    // --- 3. Time stepping loop (4th-order Runge-Kutta) ---
    for (int it = 1; it < nt; it++) {
        
        // ★デバッグ表示
        if (it <= 10 || it % 1000 == 0) {
            printf("Step %d / %d\r", it, nt);
            fflush(stdout);
        }

        // 4a. 積分因子ステップ
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf[j][i] *= exp_factor[j][i];
                nf[j][i] *= exp_factor[j][i];
            }
        }
        
        // 4b. 4次ルンゲ＝クッタ法
        adv(zetaf, nf, dx, dy, alph, nu, kap, gw1, ga1);

        // k2
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + 0.5 * dt * gw1[j][i];
                nf_temp[j][i] = nf[j][i] + 0.5 * dt * ga1[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw2, ga2);

        // k3
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + 0.5 * dt * gw2[j][i];
                nf_temp[j][i] = nf[j][i] + 0.5 * dt * ga2[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw3, ga3);

        // k4
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + dt * gw3[j][i];
                nf_temp[j][i] = nf[j][i] + dt * ga3[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw4, ga4);

        // Update
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf[j][i] += dt / 6.0 * (gw1[j][i] + 2.0 * gw2[j][i] + 2.0 * gw3[j][i] + gw4[j][i]);
                nf[j][i] += dt / 6.0 * (ga1[j][i] + 2.0 * ga2[j][i] + 2.0 * ga3[j][i] + ga4[j][i]);
            }
        }

        // ================================================================
        // [追加] 解析用バイナリデータ保存 (100ステップごと)
        // ================================================================
        if (it % 100 == 0) {
            // 保存前にポテンシャル phif を最新の zetaf から計算する
            // phi_k = -zeta_k / k^2
            for(int j = 0; j < ny; j++) {
                for(int i = 0; i < nx; i++) {
                    double k2 = -(creal(KX2[j][i]) + creal(KY2[j][i]));
                    if (k2 > 1.0e-12) {
                        phif[j][i] = -zetaf[j][i] / k2;
                    } else {
                        phif[j][i] = 0.0 + 0.0 * I;
                    }
                }
            }

            char bin_filename[256];
            sprintf(bin_filename, "%s/step_%06d.bin", dir, it);
            FILE *fp_bin = fopen(bin_filename, "wb");
            if (fp_bin) {
                // 行ごとに書き込む (nf[j] は nx個のcplx配列へのポインタ)
                for (int j = 0; j < ny; j++) {
                    fwrite(nf[j], sizeof(cplx), nx, fp_bin);
                }
                for (int j = 0; j < ny; j++) {
                    fwrite(phif[j], sizeof(cplx), nx, fp_bin);
                }
                fclose(fp_bin);
            }
        }
        // ================================================================
        
        // 4c. 履歴の保存 (メモリ上)
        if (it % isav == 0) {
            int t_idx = it / isav;
            if (t_idx < nsav) {
                save_current_history(t_idx, zetaf, nf, phif, phihst_data, nhst_data, zetahst_data);
                
                // 進捗表示
                if (t_idx % (nsav/100 + 1) == 0) { 
                      printf("\nSaved History: %d/%d (%.1f%%)\n", it, nt, (double)it / nt * 100.0);
                }
            }
        }
    }
    
    // --- 4. Cleanup ---
    save_and_cleanup(nsav, dir, phif, nf, zetaf, gw1, ga1, gw2, ga2, gw3, ga3, gw4, ga4, zetaf_temp, nf_temp, exp_factor, phihst_data, nhst_data, zetahst_data);
}
