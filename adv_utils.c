#include "MHW.h"

/**
 * @brief 渦度からポテンシャルを計算し、phi, n, zetaをフーリエ空間から実空間へ変換する。
 */
void calculate_phi_and_ifft(cplx **zetaf, cplx **nf, cplx **phif, double **phi, double **n, double **zeta) {
    // 1. Calculate phi_f
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            double k2 = creal(KX2[j][i]) + creal(KY2[j][i]);
            phif[j][i] = k2 != 0.0 ? zetaf[j][i] / (-k2) : 0.0;
        }
    }
    phif[0][0] = 0.0; // ゼロ平均を強制

    // IFFT用の作業バッファ (複素数) を確保
    cplx *tmp_out = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    fftw_plan p;

    double norm = 1.0 / (NX * NY);

    // --- Phi (Complex -> Real) ---
    p = fftw_plan_dft_2d(NY, NX, phif[0], tmp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            phi[j][i] = creal(tmp_out[j * NX + i]) * norm;
        }
    }
    fftw_destroy_plan(p);

    // --- N (Complex -> Real) ---
    p = fftw_plan_dft_2d(NY, NX, nf[0], tmp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            n[j][i] = creal(tmp_out[j * NX + i]) * norm;
        }
    }
    fftw_destroy_plan(p);

    // --- Zeta (Complex -> Real) ---
    p = fftw_plan_dft_2d(NY, NX, zetaf[0], tmp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            zeta[j][i] = creal(tmp_out[j * NX + i]) * norm;
        }
    }
    fftw_destroy_plan(p);

    fftw_free(tmp_out);
}

/**
 * @brief 帯状平均 (Zonal Averages) phiz と nz を計算する。
 */
void calculate_zonal_averages(double **phi, double **n, double dy, double LY, double *phiz, double *nz) {
    // phiz, nz は calloc でゼロ初期化されている前提
    for (int i = 0; i < NX; i++) {
        phiz[i] = 0.0;
        nz[i] = 0.0;
        for (int j = 0; j < NY; j++) {
            phiz[i] += phi[j][i];
            nz[i] += n[j][i];
        }
        // y方向に積分して長さで割る
        phiz[i] = (phiz[i] * dy) / LY;
        nz[i] = (nz[i] * dy) / LY;
    }
}

/**
 * @brief 空間微分をk空間で計算し、デエイリアシングフィルターを適用して実空間に戻す。
 */
void calculate_filtered_derivatives(cplx **phif, cplx **zetaf, cplx **nf, 
                                    cplx **phixf, cplx **phiyf, cplx **zetaxf, cplx **zetayf, cplx **nxf, cplx **nyf,
                                    double **phix, double **phiy, double **zetax, double **zetay, double **nnx, double **nny) {
    
    // 1. Calculate spatial derivatives in k-space
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            cplx filter = KXD[j][i] * KYD[j][i];
            
            phixf[j][i] = I * KX[j][i] * phif[j][i] * filter;
            phiyf[j][i] = I * KY[j][i] * phif[j][i] * filter;
            zetaxf[j][i] = I * KX[j][i] * zetaf[j][i] * filter;
            zetayf[j][i] = I * KY[j][i] * zetaf[j][i] * filter;
            nxf[j][i] = I * KX[j][i] * nf[j][i] * filter;
            nyf[j][i] = I * KY[j][i] * nf[j][i] * filter;
        }
    }
    
    // IFFT Execution using temporary buffer
    cplx *tmp_out = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    fftw_plan p;
    double norm = 1.0 / (NX * NY);

    // Helper macro for repetitive IFFT tasks
    #define EXECUTE_IFFT(in_cplx, out_double) \
        p = fftw_plan_dft_2d(NY, NX, in_cplx[0], tmp_out, FFTW_BACKWARD, FFTW_ESTIMATE); \
        fftw_execute(p); \
        for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) out_double[j][i] = creal(tmp_out[j*NX+i]) * norm; \
        fftw_destroy_plan(p);

    EXECUTE_IFFT(phixf, phix);
    EXECUTE_IFFT(phiyf, phiy);
    EXECUTE_IFFT(zetaxf, zetax);
    EXECUTE_IFFT(zetayf, zetay);
    EXECUTE_IFFT(nxf, nnx);
    EXECUTE_IFFT(nyf, nny);

    #undef EXECUTE_IFFT
    fftw_free(tmp_out);
}

/**
 * @brief 実空間でMHW方程式の右辺 (advf, advg) を計算する。
 */
void calculate_rhs_real_space(double **phi, double **n, double **zeta, double **phix, double **phiy, double **zetax, 
                              double **zetay, double **nnx, double **nny, double *phiz, double *nz,
                              double alph, double nu, double kap, double **advf, double **advg, cplx **phif) {
    
    // advg の最終項 (-kap * v_x = -kap * d(phi)/d(y)) 用の処理 (Unfiltered)
    cplx *tmp_in = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    cplx *tmp_out = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    
    // Prepare phiyf_unf in tmp_in
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            tmp_in[j * NX + i] = I * KY[j][i] * phif[j][i];
        }
    }

    fftw_plan p = fftw_plan_dft_2d(NY, NX, tmp_in, tmp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    double norm = 1.0 / (NX * NY);
    
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            double phiy_val = creal(tmp_out[j * NX + i]) * norm;

            double phiz_val = phiz[i];
            double nz_val = nz[i];
            
            // Equation for zeta (advf)
            advf[j][i] = -(phix[j][i] * zetay[j][i] - phiy[j][i] * zetax[j][i]) 
                       + alph * ((phi[j][i] - phiz_val) - (n[j][i] - nz_val)) 
                       - nu * zeta[j][i];

            // Equation for n (advg)
            advg[j][i] = -(phix[j][i] * nny[j][i] - phiy[j][i] * nnx[j][i])
                       + alph * ((phi[j][i] - phiz_val) - (n[j][i] - nz_val)) 
                       - kap * phiy_val; 
        }
    }
    
    fftw_free(tmp_in);
    fftw_free(tmp_out);
}

/**
 * @brief 実空間の右辺 (advf, advg) をフーリエ空間 (advff, advgf) に変換する。
 */
void calculate_fft_rhs(double **advf, double **advg, cplx **advff, cplx **advgf) {
    cplx *tmp_in = (cplx *)fftw_malloc(sizeof(cplx) * NX * NY);
    fftw_plan p;

    // --- advf (Real -> Complex) ---
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            tmp_in[j * NX + i] = advf[j][i]; // Imaginary part is 0.0
        }
    }
    p = fftw_plan_dft_2d(NY, NX, tmp_in, advff[0], FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    // --- advg (Real -> Complex) ---
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            tmp_in[j * NX + i] = advg[j][i];
        }
    }
    p = fftw_plan_dft_2d(NY, NX, tmp_in, advgf[0], FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    fftw_free(tmp_in);
}
