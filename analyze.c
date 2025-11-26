#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <string.h>

// --- シミュレーションの設定に合わせて変更してください ---
#define NX 256
#define NY 256
#define LX 10.0  // シミュレーションの系長Lx
#define LY 10.0  // シミュレーションの系長Ly
// --------------------------------------------------

int main(int argc, char **argv) {
    // データファイルの範囲設定（引数や固定値で指定）
    int start_step = 0;
    int end_step = 4000;   // データがある最後のステップ
    int interval = 100;    // 保存した間隔

    // メモリ確保
    int complex_size = NX * (NY / 2 + 1);
    fftw_complex *c_n = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);
    fftw_complex *c_phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);
    
    // 粒子束計算用（実空間作業領域）
    double *n_real = (double*) fftw_malloc(sizeof(double) * NX * NY);
    double *vx_real = (double*) fftw_malloc(sizeof(double) * NX * NY);
    fftw_complex *c_vx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);

    // FFTWプラン作成 (c2r: 複素 -> 実)
    fftw_plan plan_n = fftw_plan_dft_c2r_2d(NX, NY, c_n, n_real, FFTW_ESTIMATE);
    fftw_plan plan_vx = fftw_plan_dft_c2r_2d(NX, NY, c_vx, vx_real, FFTW_ESTIMATE);

    // 結果出力用ファイル
    FILE *fp_out = fopen("analysis_result.txt", "w");
    fprintf(fp_out, "# Step  E_turb  E_ZF  E_ST  Flux_mean\n");

    printf("Analysis started...\n");

    for (int step = start_step; step <= end_step; step += interval) {
        char filename[64];
        sprintf(filename, "data/step_%06d.bin", step);

        FILE *fp = fopen(filename, "rb");
        if (fp == NULL) {
            // ファイルがない場合はスキップまたは終了
            continue;
        }

        // データの読み込み
        fread(c_n, sizeof(fftw_complex), complex_size, fp);
        fread(c_phi, sizeof(fftw_complex), complex_size, fp);
        fclose(fp);

        // --- 解析計算 ---
        double Eturb = 0.0, EZF = 0.0, EST = 0.0;
        double norm = 1.0; // 必要に応じて正規化係数を調整

        // 1. エネルギー計算 (波数空間)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < NY / 2 + 1; j++) {
                int idx = i * (NY / 2 + 1) + j;
                
                // 波数の計算
                double kx, ky;
                if (i <= NX / 2) kx = (double)i * 2.0 * M_PI / LX;
                else             kx = (double)(i - NX) * 2.0 * M_PI / LX;
                ky = (double)j * 2.0 * M_PI / LY;

                double k2 = kx * kx + ky * ky;
                if (k2 == 0.0) continue; // 平均値成分は無視

                // |phi|^2, |n|^2
                double phi_sq = c_phi[idx][0]*c_phi[idx][0] + c_phi[idx][1]*c_phi[idx][1];
                double n_sq = c_n[idx][0]*c_n[idx][0] + c_n[idx][1]*c_n[idx][1];

                // 運動エネルギー (0.5 * k^2 * |phi|^2) + 内部エネルギー (0.5 * |n|^2)
                double E_k = 0.5 * (k2 * phi_sq + n_sq);

                // --- エネルギーの分類 (Pythonコードの論理に基づく) ---
                if (j == 0 && i != 0) {
                    // ky=0 (Zonal Flow / Zonal Density)
                    EZF += E_k;
                } else if (i == 0 && j != 0) {
                    // kx=0 (Streamer)
                    EST += E_k;
                } else {
                    // Turbulence
                    Eturb += E_k;
                }
                
                // 2. 粒子束のための速度場の準備 (v_x = -ik_y * phi)
                // -i * ky * (Re + iIm) = ky*Im - i*ky*Re
                c_vx[idx][0] =  ky * c_phi[idx][1];
                c_vx[idx][1] = -ky * c_phi[idx][0];
            }
        }

        // 3. 粒子束の計算 (実空間へ変換)
        fftw_execute(plan_n);   // c_n -> n_real
        fftw_execute(plan_vx);  // c_vx -> vx_real

        double flux_sum = 0.0;
        double fft_norm = 1.0 / ((double)NX * NY); // FFTW逆変換の正規化

        for (int k = 0; k < NX * NY; k++) {
            // FFTWの出力はスケーリングされていないため、ここで正規化
            double n_val = n_real[k] * fft_norm;
            double vx_val = vx_real[k] * fft_norm;
            flux_sum += n_val * vx_val;
        }
        double flux_mean = flux_sum / (NX * NY); // 空間平均

        // 結果を書き出し
        fprintf(fp_out, "%d %e %e %e %e\n", step, Eturb, EZF, EST, flux_mean);
        
        if(step % 1000 == 0) printf("Processed step %d\n", step);
    }

    fclose(fp_out);
    fftw_destroy_plan(plan_n);
    fftw_destroy_plan(plan_vx);
    fftw_free(c_n);
    fftw_free(c_phi);
    fftw_free(c_vx);
    fftw_free(n_real);
    fftw_free(vx_real);

    printf("Done. Results saved to analysis_result.txt\n");
    return 0;
}
