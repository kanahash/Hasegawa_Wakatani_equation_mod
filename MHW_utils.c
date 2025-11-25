#include "MHW.h"

// --- Global Variable Definition ---
// MHW.h で extern 宣言されたグローバル変数の実体定義
int NX, NY;
double LX, LY;
int shot_no = 13; 
cplx **KX, **KY, **KX2, **KY2, **KXD, **KYD;

// --- Memory Allocation/Deallocation Implementations ---
/**
 * @brief 2次元 double 配列を連続メモリで確保する。
 */
double **alloc_2d_double(int rows, int cols) {
    double *data = (double *)malloc(rows * cols * sizeof(double));
    double **array = (double **)malloc(rows * sizeof(double *));
    if (data == NULL || array == NULL) {
        if (data != NULL) free(data);
        if (array != NULL) free(array);
        fprintf(stderr, "Memory allocation failed for double array.\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        array[i] = &(data[i * cols]);
    }
    return array;
}

/**
 * @brief 2次元 double 配列を解放する。
 */
void free_2d_double(double **array) {
    if (array != NULL) {
        if (array[0] != NULL) free(array[0]);
        free(array);
    }
}

/**
 * @brief 2次元 cplx (FFTW対応) 配列を連続メモリで確保する。
 */
cplx **alloc_2d_cplx(int rows, int cols) {
    cplx *data = (cplx *)fftw_malloc(rows * cols * sizeof(cplx));
    cplx **array = (cplx **)malloc(rows * sizeof(cplx *));
    if (data == NULL || array == NULL) {
        if (data != NULL) fftw_free(data);
        if (array != NULL) free(array);
        fprintf(stderr, "Memory allocation failed for cplx array.\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        array[i] = &(data[i * cols]);
    }
    return array;
}

/**
 * @brief 2次元 cplx (FFTW対応) 配列を解放する。
 */
void free_2d_cplx(cplx **array) {
    if (array != NULL) {
        if (array[0] != NULL) fftw_free(array[0]);
        free(array);
    }
}
// --- End Memory Allocation/Deallocation Implementations ---

/**
 * @brief MHW方程式のメインシミュレーションを実行する。
 */
void MHW(int nx, int ny, double lx, double ly, int nt, double dt, double kap, double alph, double mu, double nu, double **phi_init, double **n_init, int isav, const char *dir) {
    // グローバル変数の設定
    NX = nx; NY = ny; LX = lx; LY = ly;
    double dx = lx / nx;
    double dy = ly / ny;
    int nsav = nt / isav;

    // --- 1. Allocate main state buffers and RK4 buffers ---
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
    size_t hist_size = nsav * nx * ny;
    double *phihst_data = (double *)calloc(hist_size, sizeof(double));
    double *nhst_data = (double *)calloc(hist_size, sizeof(double));
    double *zetahst_data = (double *)calloc(hist_size, sizeof(double));
    
    // エラーチェック (簡略化): 実際のコードでは各 alloc_2d_cplx/double の NULL チェックが必要です

    // --- 2. Setup grid, initialize state, and calculate integrating factor ---
    setup_grid_and_wavenumbers(nx, ny, lx, ly); // グローバル波数配列を設定
    
    // 初期状態の計算と t=0 の履歴保存
    initialize_state_and_history(nx, ny, nsav, phi_init, n_init, phif, nf, zetaf, phihst_data, nhst_data, zetahst_data);
    
    // 積分因子 exp_factor の計算
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double k2 = creal(KX2[j][i]) + creal(KY2[j][i]);
            exp_factor[j][i] = cexp(-mu * k2 * k2 * dt);
        }
    }

    printf("Starting MHW simulation for %d timesteps (saved steps: %d)...\n", nt, nsav);

    // --- 3. Time stepping loop (4th-order Runge-Kutta) ---
    for (int it = 1; it < nt; it++) {
        // 4a. 積分因子ステップ (拡散項の厳密な時間発展)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf[j][i] *= exp_factor[j][i];
                nf[j][i] *= exp_factor[j][i];
            }
        }
        
        // 4b. 4次ルンゲ＝クッタ法 (非線形項の積分)
        
        // k1 = adv(zetaf, nf)
        adv(zetaf, nf, dx, dy, alph, nu, kap, gw1, ga1);

        // k2 = adv(zetaf + 0.5*dt*k1, nf + 0.5*dt*k1_n)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + 0.5 * dt * gw1[j][i];
                nf_temp[j][i] = nf[j][i] + 0.5 * dt * ga1[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw2, ga2);

        // k3 = adv(zetaf + 0.5*dt*k2, nf + 0.5*dt*k2_n)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + 0.5 * dt * gw2[j][i];
                nf_temp[j][i] = nf[j][i] + 0.5 * dt * ga2[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw3, ga3);

        // k4 = adv(zetaf + dt*k3, nf + dt*k3_n)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf_temp[j][i] = zetaf[j][i] + dt * gw3[j][i];
                nf_temp[j][i] = nf[j][i] + dt * ga3[j][i];
            }
        }
        adv(zetaf_temp, nf_temp, dx, dy, alph, nu, kap, gw4, ga4);

        // 最終的な更新: u^{n+1} = u^n + dt/6 * (k1 + 2k2 + 2k3 + k4)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                zetaf[j][i] += dt / 6.0 * (gw1[j][i] + 2.0 * gw2[j][i] + 2.0 * gw3[j][i] + gw4[j][i]);
                nf[j][i] += dt / 6.0 * (ga1[j][i] + 2.0 * ga2[j][i] + 2.0 * ga3[j][i] + ga4[j][i]);
            }
        }
        
        // 4c. 履歴の保存
        if (it % isav == 0) {
            int t_idx = it / isav;
            save_current_history(t_idx, zetaf, nf, phif, phihst_data, nhst_data, zetahst_data);
            
            if (t_idx % (nsav/10 + 1) == 0) {
                 printf("Timestep: %d/%d (%.1f%%)\n", it, nt, (double)it / nt * 100.0);
            }
        }
    }
    
    // --- 4. 最終保存とクリーンアップ ---
    save_and_cleanup(nsav, dir, phif, nf, zetaf, gw1, ga1, gw2, ga2, gw3, ga3, gw4, ga4, zetaf_temp, nf_temp, exp_factor, phihst_data, nhst_data, zetahst_data);
}
