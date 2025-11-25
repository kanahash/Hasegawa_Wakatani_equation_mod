#include "MHW.h"

// --- Global Variable Definition ---
int NX, NY;
double LX, LY;
int shot_no = 13; 
cplx **KX, **KY, **KX2, **KY2, **KXD, **KYD;

// --- Helper Functions Implementation ---

/**
 * @brief 波数空間のグリッドとデエイリアシングフィルタを設定する
 */
void setup_grid_and_wavenumbers(int nx, int ny, double lx, double ly) {
    // 1. メモリ確保 (ここが抜けていたためクラッシュしていました)
    KX = alloc_2d_cplx(ny, nx);
    KY = alloc_2d_cplx(ny, nx);
    KX2 = alloc_2d_cplx(ny, nx);
    KY2 = alloc_2d_cplx(ny, nx);
    KXD = alloc_2d_cplx(ny, nx); // 追加
    KYD = alloc_2d_cplx(ny, nx); // 追加

    double kx_val, ky_val;
    
    // デエイリアシング (2/3ルール) のカットオフ波数
    // int kx_cut = (int)(2.0 / 3.0 * (nx / 2));
    // int ky_cut = (int)(2.0 / 3.0 * (ny / 2));
    // 簡易的に全モード通す場合はフィルタを1.0にしますが、通常は非線形項の安定化のため2/3ルールを使います。
    // ここでは単純化のため、まずは「フィルタなし(全て1.0)」または「2/3ルール」を実装します。
    
    for (int j = 0; j < ny; j++) {
        // FFTW frequency ordering: 0, 1, ..., N/2-1, -N/2, ..., -1
        int m = (j <= ny / 2) ? j : j - ny;
        ky_val = 2.0 * M_PI * m / ly;

        // Y方向のデエイリアシング判定 (2/3 rule)
        double filter_y = 1.0;
        if (abs(m) > (ny / 3)) filter_y = 0.0;

        for (int i = 0; i < nx; i++) {
            int l = (i <= nx / 2) ? i : i - nx;
            kx_val = 2.0 * M_PI * l / lx;

            KX[j][i] = kx_val;
            KY[j][i] = ky_val;
            
            // Laplacian (-k^2)
            KX2[j][i] = -(kx_val * kx_val);
            KY2[j][i] = -(ky_val * ky_val);

            // X方向のデエイリアシング判定 (2/3 rule)
            double filter_x = 1.0;
            if (abs(l) > (nx / 3)) filter_x = 0.0;

            // フィルタ配列の設定
            KXD[j][i] = filter_x;
            KYD[j][i] = filter_y;
        }
    }
}

/**
 * @brief 初期状態をFFTし、渦度(Zeta)を計算し、t=0の履歴を保存する
 */
void initialize_state_and_history(int nx, int ny, int nsav, 
                                  double **phi_init, double **n_init, 
                                  cplx **phif, cplx **nf, cplx **zetaf, 
                                  double *phihst_data, double *nhst_data, double *zetahst_data) {
    (void)nsav; 

    cplx *in_buf = (cplx *)fftw_malloc(sizeof(cplx) * nx * ny);
    cplx *out_buf = (cplx *)fftw_malloc(sizeof(cplx) * nx * ny);
    
    fftw_plan p = fftw_plan_dft_2d(ny, nx, in_buf, out_buf, FFTW_FORWARD, FFTW_ESTIMATE);

    // --- Phi ---
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            in_buf[j * nx + i] = phi_init[j][i];
        }
    }
    fftw_execute(p);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            phif[j][i] = out_buf[j * nx + i] / (double)(nx * ny);
        }
    }

    // --- N ---
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            in_buf[j * nx + i] = n_init[j][i];
        }
    }
    fftw_execute(p);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            nf[j][i] = out_buf[j * nx + i] / (double)(nx * ny);
        }
    }

    // --- Zeta (-k^2 * phi) ---
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double laplacian_k = creal(KX2[j][i]) + creal(KY2[j][i]);
            zetaf[j][i] = laplacian_k * phif[j][i];
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in_buf);
    fftw_free(out_buf);

    save_current_history(0, zetaf, nf, phif, phihst_data, nhst_data, zetahst_data);
}

/**
 * @brief 現在のステップの状態を履歴配列に保存する
 */
void save_current_history(int t_idx, cplx **zetaf, cplx **nf, cplx **phif, 
                          double *phihst_data, double *nhst_data, double *zetahst_data) {
    
    cplx *in_buf = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    cplx *out_buf = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    fftw_plan p = fftw_plan_dft_2d(NY, NX, in_buf, out_buf, FFTW_BACKWARD, FFTW_ESTIMATE);

    size_t offset = (size_t)t_idx * NX * NY;

    // --- Phi ---
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            in_buf[j * NX + i] = phif[j][i];
        }
    }
    fftw_execute(p);
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            phihst_data[offset + j * NX + i] = creal(out_buf[j * NX + i]);
        }
    }

    // --- N ---
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            in_buf[j * NX + i] = nf[j][i];
        }
    }
    fftw_execute(p);
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            nhst_data[offset + j * NX + i] = creal(out_buf[j * NX + i]);
        }
    }

    // --- Zeta ---
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            in_buf[j * NX + i] = zetaf[j][i];
        }
    }
    fftw_execute(p);
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            zetahst_data[offset + j * NX + i] = creal(out_buf[j * NX + i]);
        }
    }

    fftw_destroy_plan(p);
    fftw_free(in_buf);
    fftw_free(out_buf);
}

// --- Save and Cleanup ---

void save_and_cleanup(int nsav, const char *dir,
                      cplx **phif, cplx **nf, cplx **zetaf,
                      cplx **gw1, cplx **ga1, cplx **gw2, cplx **ga2,
                      cplx **gw3, cplx **ga3, cplx **gw4, cplx **ga4,
                      cplx **zetaf_temp, cplx **nf_temp, cplx **exp_factor,
                      double *phihst_data, double *nhst_data, double *zetahst_data) {
    
    char filepath[256];
    FILE *fp;
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        #ifdef __linux__
        mkdir(dir, 0700);
        #else
        _mkdir(dir);
        #endif
    }

    snprintf(filepath, sizeof(filepath), "%s/phi_history.dat", dir);
    fp = fopen(filepath, "wb");
    if (fp) {
        fwrite(phihst_data, sizeof(double), (size_t)nsav * NX * NY, fp);
        fclose(fp);
        printf("Saved: %s\n", filepath);
    }
    
    snprintf(filepath, sizeof(filepath), "%s/n_history.dat", dir);
    fp = fopen(filepath, "wb");
    if (fp) {
        fwrite(nhst_data, sizeof(double), (size_t)nsav * NX * NY, fp);
        fclose(fp);
        printf("Saved: %s\n", filepath);
    }

    snprintf(filepath, sizeof(filepath), "%s/zeta_history.dat", dir);
    fp = fopen(filepath, "wb");
    if (fp) {
        fwrite(zetahst_data, sizeof(double), (size_t)nsav * NX * NY, fp);
        fclose(fp);
        printf("Saved: %s\n", filepath);
    }

    // Cleanup local buffers
    free_2d_cplx(phif); free_2d_cplx(nf); free_2d_cplx(zetaf);
    free_2d_cplx(gw1); free_2d_cplx(ga1);
    free_2d_cplx(gw2); free_2d_cplx(ga2);
    free_2d_cplx(gw3); free_2d_cplx(ga3);
    free_2d_cplx(gw4); free_2d_cplx(ga4);
    free_2d_cplx(zetaf_temp); free_2d_cplx(nf_temp); free_2d_cplx(exp_factor);
    
    // Cleanup globals (KXD, KYD の解放も追加)
    free_2d_cplx(KX); free_2d_cplx(KY);
    free_2d_cplx(KX2); free_2d_cplx(KY2);
    free_2d_cplx(KXD); free_2d_cplx(KYD); // 追加

    if (phihst_data) free(phihst_data);
    if (nhst_data) free(nhst_data);
    if (zetahst_data) free(zetahst_data);

    printf("Simulation cleanup completed.\n");
}
