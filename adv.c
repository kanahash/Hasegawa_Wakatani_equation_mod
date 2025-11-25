#include "MHW.h"

void adv(cplx **zetaf, cplx **nf, double dx, double dy, double alph, double nu, double kap, cplx **advff, cplx **advgf) {
    // 未使用パラメータの抑制
    (void)dx; 
    // dyはcalculate_zonal_averagesで使用

    // テンポラリ配列の確保 (double型)
    cplx **phif = alloc_2d_cplx(NY, NX);
    double **phi = alloc_2d_double(NY, NX);
    double **n = alloc_2d_double(NY, NX);
    double **zeta = alloc_2d_double(NY, NX);

    double **phix = alloc_2d_double(NY, NX);
    double **phiy = alloc_2d_double(NY, NX);
    double **zetax = alloc_2d_double(NY, NX);
    double **zetay = alloc_2d_double(NY, NX);
    double **nnx = alloc_2d_double(NY, NX);
    double **nny = alloc_2d_double(NY, NX);
    
    double **advf = alloc_2d_double(NY, NX);
    double **advg = alloc_2d_double(NY, NX);

    cplx **phixf = alloc_2d_cplx(NY, NX);
    cplx **phiyf = alloc_2d_cplx(NY, NX);
    cplx **zetaxf = alloc_2d_cplx(NY, NX);
    cplx **zetayf = alloc_2d_cplx(NY, NX);
    cplx **nxf = alloc_2d_cplx(NY, NX);
    cplx **nyf = alloc_2d_cplx(NY, NX);

    double *phiz = (double *)calloc(NX, sizeof(double));
    double *nz = (double *)calloc(NX, sizeof(double));

    // 1. phiの計算と、phi, n, zetaの逆フーリエ変換 (IFFT)
    calculate_phi_and_ifft(zetaf, nf, phif, phi, n, zeta);

    // 2. 帯状平均 (Zonal Average) の計算
    calculate_zonal_averages(phi, n, dy, LY, phiz, nz);

    // 3. 空間微分の計算 (FFT -> IFFT & De-aliasing)
    calculate_filtered_derivatives(phif, zetaf, nf, 
                                   phixf, phiyf, zetaxf, zetayf, nxf, nyf,
                                   phix, phiy, zetax, zetay, nnx, nny);

    // 4. 実空間で時間発展の右辺 (RHS) を計算
    calculate_rhs_real_space(phi, n, zeta, phix, phiy, zetax, zetay, nnx, nny, 
                             phiz, nz, alph, nu, kap, advf, advg, phif);

    // 5. 右辺 (RHS) のフーリエ変換 (FFT)
    calculate_fft_rhs(advf, advg, advff, advgf);
    
    // クリーンアップ
    free(phiz); free(nz);
    free_2d_cplx(phif); free_2d_double(phi); free_2d_double(n); free_2d_double(zeta);
    free_2d_double(phix); free_2d_double(phiy); free_2d_double(zetax); free_2d_double(zetay);
    free_2d_double(nnx); free_2d_double(nny); free_2d_double(advf); free_2d_double(advg);
    free_2d_cplx(phixf); free_2d_cplx(phiyf); free_2d_cplx(zetaxf); free_2d_cplx(zetayf);
    free_2d_cplx(nxf); free_2d_cplx(nyf);
}
