#include "MHW.h"

// --- Global Variable Definition ---
int NX, NY;
double LX, LY;
int shot_no = 13; 
cplx **KX, **KY, **KX2, **KY2, **KXD, **KYD;

// --- Helper Functions for FFT (Internal use) ---
// Note: 2d_array_custom.c の関数を使用
// MHW.h でプロトタイプ宣言されている前提

// --- Implementation of Missing Functions ---

/**
 * @brief 波数空間のグリッドを設定する (FFTWの順序に従う)
 */
void setup_grid_and_wavenumbers(int nx, int ny, double lx, double ly) {
    // グローバル変数のメモリ確保
    KX = alloc_2d_cplx(ny, nx);
    KY = alloc_2d_cplx(ny, nx);
    KX2 = alloc_2d_cplx(ny, nx);
    KY2 = alloc_2d_cplx(ny, nx);
    // 必要に応じて KXD, KYD もここで確保（デエイリアシング用など）
    // 今回は単純なシミュレーション用として基本波数のみ設定

    double kx_val, ky_val;
    for (int j = 0; j < ny; j++) {
        // FFTWの周波数順序: 0, 1, ..., N/2-1, -N/2, ..., -1
        int m = (j <= ny / 2) ? j : j - ny;
        ky_val = 2.0 * M_PI * m / ly;

        for (int i = 0; i < nx; i++) {
            int l = (i <= nx / 2) ? i : i - nx;
            kx_val = 2.0 * M_PI * l / lx;

            KX[j][i] = kx_val;
            KY[j][i] = ky_val;
            
            // ラプラシアン用 (-k^2)
            KX2[j][i] = -kx_val * kx_val;
            KY2[j][i] = -ky_val * ky_val;
        }
    }
}

/**
 * @brief 初期状態をフーリエ変換して設定し、t=0の履歴を保存する
 */
void initialize_state_and_history(int nx, int ny, int nsav, 
                                  double **phi_init, double **n_init, 
                                  cplx **phif, cplx **nf, cplx **zetaf, 
                                  double *phihst_data, double *nhst_data, double *zetahst_data) {
    (void)nsav; // unused warning suppression if needed

    // 1. Double (Real) -> Complex 変換のためのバッファ
    cplx *in_buf = (cplx *)fftw_malloc(sizeof(cplx) * nx * ny);
    cplx *out_buf = (cplx *)fftw_malloc(sizeof(cplx) * nx * ny);
    fftw_plan p = fftw_plan_dft_2d(ny, nx, in_buf, out_buf, FFTW_FORWARD, FFTW_ESTIMATE);

    // --- Phi の変換 ---
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            in_buf[j * nx + i] = phi_init[j][i];
        }
    }
    fftw_execute(p);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            phif[j][i] = out_buf[j * nx + i] / (double)(nx * ny); // 正規化
        }
    }

    // --- N の変換 ---
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

    // --- Zeta (Vorticity) の計算: zeta_k = -k^2 * phi_k ---
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double k2 = -(creal(KX2[j][i]) + creal(KY2[j][i])); // k^2
            // Laplacian operation in spectral space: -k^2
            zetaf[j][i] = -k2 * phif[j][i];
        }
    }

    // クリーンアップ
    fftw_destroy_plan(p);
    fftw_free(in_buf);
    fftw_free(out_buf);

    // t=0 の履歴保存 (t_idx = 0)
    save_current_history(0, zetaf, nf, phif, phihst_data, nhst_data, zetahst_data);
}


/**
 * @brief 現在のステップのデータを実空間に戻して履歴配列に保存する
 */
void save_current_history(int t_idx, cplx **zetaf, cplx **nf, cplx **phif, 
                          double *phihst_data, double *nhst_data, double *zetahst_data) {
    
    // 逆変換用のプランとバッファ
    cplx *in_buf = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    cplx *out_buf = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    fftw_plan p = fftw_plan_dft_2d(NY, NX, in_buf, out_buf, FFTW_BACKWARD, FFTW_ESTIMATE);

    size_t offset = (size_t)t_idx * NX * NY;

    // --- Phi 保存 ---
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            in_buf[j * NX + i] = phif[j][i];
        }
    }
    fftw_execute(p);
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            phihst_data[offset + j * NX + i] = creal(out_buf[j * NX + i]); 
            // FFTW Backwardは非正規化なので、Initializeで割っていればここはそのまま、
            // もしくは往復でサイズ倍になるのを考慮。ここではInitializeで割っているのでOKと仮定。
        }
    }

    // --- N 保存 ---
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

    // --- Zeta 保存 ---
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

// --- Save and Cleanup (Previous Implementation) ---

void save_and_cleanup(int nsav, const char *dir,
                      cplx **phif, cplx **nf, cplx **zetaf,
                      cplx **gw1, cplx **ga1, cplx **gw2, cplx **ga2,
                      cplx **gw3, cplx **ga3, cplx **gw4, cplx **ga4,
                      cplx **zetaf_temp, cplx **nf_temp, cplx **exp_factor,
                      double *phihst_data, double *nhst_data, double *zetahst_data) {
    
    char filepath[256];
    FILE *fp;

    // ディレクトリが存在するか確認 (簡易的)
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        #ifdef __linux__
        mkdir(dir, 0700);
        #else
        _mkdir(dir); // Windows environment
        #endif
    }

    // Phi History の保存
    snprintf(filepath, sizeof(filepath), "%s/phi_history.dat", dir);
    fp = fopen(filepath, "wb");
    if (fp) {
        size_t total_elements = (size_t)nsav * NX * NY; 
        fwrite(phihst_data, sizeof(double), total_elements, fp);
        fclose(fp);
        printf("Saved: %s\n", filepath);
    } else {
        fprintf(stderr, "Error: Could not open file %s for writing.\n", filepath);
    }
    
    // N History の保存
    snprintf(filepath, sizeof(filepath), "%s/n_history.dat", dir);
    fp = fopen(filepath, "wb");
    if (fp) {
        size_t total_elements = (size_t)nsav * NX * NY; 
        fwrite(nhst_data, sizeof(double), total_elements, fp);
        fclose(fp);
        printf("Saved: %s\n", filepath);
    }

    // Zeta History の保存
    snprintf(filepath, sizeof(filepath), "%s/zeta_history.dat", dir);
    fp = fopen(filepath, "wb");
    if (fp) {
        size_t total_elements = (size_t)nsav * NX * NY; 
        fwrite(zetahst_data, sizeof(double), total_elements, fp);
        fclose(fp);
        printf("Saved: %s\n", filepath);
    }

    // メモリ解放 (2d_array_custom.c の関数を使用)
    free_2d_cplx(phif); free_2d_cplx(nf); free_2d_cplx(zetaf);
    free_2d_cplx(gw1); free_2d_cplx(ga1);
    free_2d_cplx(gw2); free_2d_cplx(ga2);
    free_2d_cplx(gw3); free_2d_cplx(ga3);
    free_2d_cplx(gw4); free_2d_cplx(ga4);
    free_2d_cplx(zetaf_temp); free_2d_cplx(nf_temp); free_2d_cplx(exp_factor);
    
    // グローバル変数の解放
    free_2d_cplx(KX); free_2d_cplx(KY);
    free_2d_cplx(KX2); free_2d_cplx(KY2);

    if (phihst_data != NULL) free(phihst_data);
    if (nhst_data != NULL) free(nhst_data);
    if (zetahst_data != NULL) free(zetahst_data);

    printf("Simulation cleanup completed.\n");
}
