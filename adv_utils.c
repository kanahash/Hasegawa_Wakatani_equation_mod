#include "MHW.h"

/**
 * @brief 渦度からポテンシャルを計算し、phi, n, zetaをフーリエ空間から実空間へ変換する。
 */
void calculate_phi_and_ifft(cplx **zetaf, cplx **nf, cplx **phif, double **phi, double **n, double **zeta) {
    fftw_plan p_ifft_phi, p_ifft_n, p_ifft_zeta;

    // 1. Calculate phi_f and its inverse
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            double k2 = creal(KX2[j][i]) + creal(KY2[j][i]);
            phif[j][i] = k2 != 0.0 ? zetaf[j][i] / (-k2) : 0.0;
        }
    }
    phif[0][0] = 0.0; // ゼロ平均を強制
    
    // IFFT Plan & Execute
    p_ifft_phi = fftw_plan_dft_2d(NY, NX, phif[0], (fftw_complex *)phi[0], FFTW_BACKWARD, FFTW_ESTIMATE);
    p_ifft_n = fftw_plan_dft_2d(NY, NX, nf[0], (fftw_complex *)n[0], FFTW_BACKWARD, FFTW_ESTIMATE);
    p_ifft_zeta = fftw_plan_dft_2d(NY, NX, zetaf[0], (fftw_complex *)zeta[0], FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(p_ifft_phi);
    fftw_execute(p_ifft_n);
    fftw_execute(p_ifft_zeta);

    // Normalize IFFT results (FFTW convention)
    double norm = 1.0 / (NX * NY);
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            phi[j][i] = creal(phi[j][i]) * norm; // IFFT結果は複素数バッファに入っているため実部を取得
            n[j][i] = creal(n[j][i]) * norm;
            zeta[j][i] = creal(zeta[j][i]) * norm;
        }
    }

    fftw_destroy_plan(p_ifft_phi);
    fftw_destroy_plan(p_ifft_n);
    fftw_destroy_plan(p_ifft_zeta);
}

/**
 * @brief 帯状平均 (Zonal Averages) phiz と nz を計算する。
 */
void calculate_zonal_averages(double **phi, double **n, double dy, double LY, double *phiz, double *nz) {
    // phiz, nz は calloc でゼロ初期化されている前提
    for (int i = 0; i < NX; i++) {
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
    fftw_plan p_ifft_phix, p_ifft_phiy, p_ifft_zetax, p_ifft_zetay, p_ifft_nx, p_ifft_ny;
    double norm = 1.0 / (NX * NY);
    
    // 3. Calculate spatial derivatives in k-space and apply de-aliasing filter
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            cplx filter = KXD[j][i] * KYD[j][i];
            
            // k-space multiplication
            phixf[j][i] = I * KX[j][i] * phif[j][i] * filter;
            phiyf[j][i] = I * KY[j][i] * phif[j][i] * filter;
            zetaxf[j][i] = I * KX[j][i] * zetaf[j][i] * filter;
            zetayf[j][i] = I * KY[j][i] * zetaf[j][i] * filter;
            nxf[j][i] = I * KX[j][i] * nf[j][i] * filter;
            nyf[j][i] = I * KY[j][i] * nf[j][i] * filter;
        }
    }
    
    // IFFT Plan & Execute for Derivatives
    p_ifft_phix = fftw_plan_dft_2d(NY, NX, phixf[0], (fftw_complex *)phix[0], FFTW_BACKWARD, FFTW_ESTIMATE);
    p_ifft_phiy = fftw_plan_dft_2d(NY, NX, phiyf[0], (fftw_complex *)phiy[0], FFTW_BACKWARD, FFTW_ESTIMATE);
    p_ifft_zetax = fftw_plan_dft_2d(NY, NX, zetaxf[0], (fftw_complex *)zetax[0], FFTW_BACKWARD, FFTW_ESTIMATE);
    p_ifft_zetay = fftw_plan_dft_2d(NY, NX, zetayf[0], (fftw_complex *)zetay[0], FFTW_BACKWARD, FFTW_ESTIMATE);
    p_ifft_nx = fftw_plan_dft_2d(NY, NX, nxf[0], (fftw_complex *)nnx[0], FFTW_BACKWARD, FFTW_ESTIMATE);
    p_ifft_ny = fftw_plan_dft_2d(NY, NX, nyf[0], (fftw_complex *)nny[0], FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(p_ifft_phix);
    fftw_execute(p_ifft_phiy);
    fftw_execute(p_ifft_zetax);
    fftw_execute(p_ifft_zetay);
    fftw_execute(p_ifft_nx);
    fftw_execute(p_ifft_ny);

    // Normalize IFFT results
    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            phix[j][i] = creal(phix[j][i]) * norm;
            phiy[j][i] = creal(phiy[j][i]) * norm;
            zetax[j][i] = creal(zetax[j][i]) * norm;
            zetay[j][i] = creal(zetay[j][i]) * norm;
            nnx[j][i] = creal(nnx[j][i]) * norm;
            nny[j][i] = creal(nny[j][i]) * norm;
        }
    }
    fftw_destroy_plan(p_ifft_phix);
    fftw_destroy_plan(p_ifft_phiy);
    fftw_destroy_plan(p_ifft_zetax);
    fftw_destroy_plan(p_ifft_zetay);
    fftw_destroy_plan(p_ifft_nx);
    fftw_destroy_plan(p_ifft_ny);
}

/**
 * @brief 実空間でMHW方程式の右辺 (advf, advg) を計算する。
 */
void calculate_rhs_real_space(double **phi, double **n, double **zeta, double **phix, double **phiy, double **zetax, 
                              double **zetay, double **nnx, double **nny, double *phiz, double *nz,
                              double alph, double nu, double kap, double **advf, double **advg, cplx **phif) {
    
    // advg の最終項 (-kap * v_x = -kap * d(phi)/d(y)) 用の処理
    // この項だけデエイリアシングフィルターがかからないため、別途計算が必要
    cplx **phiyf_unf = alloc_2d_cplx(NY, NX);
    double **phiy_unf = alloc_2d_double(NY, NX);
    
    for (int r = 0; r < NY; r++) {
        for (int c = 0; c < NX; c++) {
            // Unfiltered phiyf: I * KY * phif
            phiyf_unf[r][c] = I * KY[r][c] * phif[r][c];
        }
    }
    fftw_plan p_ifft_phiy_unf = fftw_plan_dft_2d(NY, NX, phiyf_unf[0], (fftw_complex *)phiy_unf[0], FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_ifft_phiy_unf);
    fftw_destroy_plan(p_ifft_phiy_unf);
    double norm = 1.0 / (NX * NY);

    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            double phiz_val = phiz[i];
            double nz_val = nz[i];
            
            // Equation for zeta (advf):
            // Non-linear: -(phix*zetay - phiy*zetax)
            // Linear (coupling + dissipation): + alph*((phi-phiz)-(n-nz)) - nu*zeta
            advf[j][i] = -(phix[j][i] * zetay[j][i] - phiy[j][i] * zetax[j][i]) 
                         + alph * ((phi[j][i] - phiz_val) - (n[j][i] - nz_val)) 
                         - nu * zeta[j][i];

            // Equation for n (advg):
            // Non-linear: -(phix*nny - phiy*nnx)
            // Linear (coupling + instability): + alph*((phi-phiz)-(n-nz)) - kap * d(phi)/d(y)
            advg[j][i] = -(phix[j][i] * nny[j][i] - phiy[j][i] * nnx[j][i])
                         + alph * ((phi[j][i] - phiz_val) - (n[j][i] - nz_val)) 
                         - kap * (creal(phiy_unf[j][i]) * norm); // -kap * v_x (unfiltered)
        }
    }
    
    free_2d_cplx(phiyf_unf);
    free_2d_double(phiy_unf);
}

/**
 * @brief 実空間の右辺 (advf, advg) をフーリエ空間 (advff, advgf) に変換する。
 */
void calculate_fft_rhs(double **advf, double **advg, cplx **advff, cplx **advgf) {
    fftw_plan p_fft_advf, p_fft_advg;

    // 5. Calculate FFT of advf and advg
    // FFTW plan: input is real, output is complex (but we pass double** as fftw_complex* for simplicity)
    p_fft_advf = fftw_plan_dft_2d(NY, NX, (fftw_complex *)advf[0], advff[0], FFTW_FORWARD, FFTW_ESTIMATE);
    p_fft_advg = fftw_plan_dft_2d(NY, NX, (fftw_complex *)advg[0], advgf[0], FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p_fft_advf);
    fftw_execute(p_fft_advg);
    
    // Cleanup
    fftw_destroy_plan(p_fft_advf);
    fftw_destroy_plan(p_fft_advg);
}
