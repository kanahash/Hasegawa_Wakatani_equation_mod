#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>
#include <sys/stat.h>

// MHW.h の定義に合わせる
typedef double complex cplx;

// パラメータ (main.c と合わせる)
#define NX 256
#define NY 256
#define LX 50.0
#define LY 50.0
#define NT 4000
#define SAVE_INTERVAL 100

int main() {
    // データサイズ
    int y_dim = NY / 2 + 1;
    int complex_size = NX * y_dim;

    // メモリ確保 (1次元配列として確保して読み込むのが楽です)
    cplx *data_buffer = (cplx*) malloc(sizeof(cplx) * complex_size * 2); // n と phi の2つ分
    
    // 解析用の一時配列
    cplx *n_k   = (cplx*) fftw_malloc(sizeof(cplx) * complex_size);
    cplx *phi_k = (cplx*) fftw_malloc(sizeof(cplx) * complex_size);
    cplx *vx_k  = (cplx*) fftw_malloc(sizeof(cplx) * complex_size);
    
    // 実空間用
    double *n_real  = (double*) fftw_malloc(sizeof(double) * NX * NY);
    double *vx_real = (double*) fftw_malloc(sizeof(double) * NX * NY);

    // FFTWプラン (c2r: 複素 -> 実)
    fftw_plan plan_n  = fftw_plan_dft_c2r_2d(NX, NY, n_k, n_real, FFTW_ESTIMATE);
    fftw_plan plan_vx = fftw_plan_dft_c2r_2d(NX, NY, vx_k, vx_real, FFTW_ESTIMATE);

    // 出力ファイル
    FILE *fp_out = fopen("analysis_result.txt", "w");
    fprintf(fp_out, "step E_turb E_ZF E_ST Flux_mean\n");

    printf("Starting analysis...\n");

    for (int step = 0; step <= NT; step += SAVE_INTERVAL) {
        char filename[256];
        sprintf(filename, "data/step_%06d.bin", step);

        FILE *fp = fopen(filename, "rb");
        if (!fp) continue;

        // データの読み込み (n, phi の順で保存されている)
        // MHW.cで1行ずつ書きましたが、バイナリ上は連続しているので一気に読めます
        fread(n_k, sizeof(cplx), complex_size, fp);
        fread(phi_k, sizeof(cplx), complex_size, fp);
        fclose(fp);

        // --- 1. エネルギー計算 (波数空間) ---
        double Eturb = 0.0, EZF = 0.0, EST = 0.0;
        
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < y_dim; j++) {
                int idx = i * y_dim + j;
                
                // 波数 kx, ky の計算
                double kx = (i <= NX/2) ? (double)i * 2*M_PI/LX : (double)(i-NX) * 2*M_PI/LX;
                double ky = (double)j * 2*M_PI/LY;
                double k2 = kx*kx + ky*ky;

                if (k2 == 0) continue;

                // エネルギー成分 = 0.5 * (k^2|phi|^2 + |n|^2)
                // cabs() は複素数の絶対値を返す関数
                double p_sq = creal(phi_k[idx])*creal(phi_k[idx]) + cimag(phi_k[idx])*cimag(phi_k[idx]);
                double n_sq = creal(n_k[idx])*creal(n_k[idx]) + cimag(n_k[idx])*cimag(n_k[idx]);
                
                // FFTWの仕様上、係数が大きくなっている場合の補正が必要ですが
                // ここでは相対比較のためそのまま加算します
                double E_mode = 0.5 * (k2 * p_sq + n_sq);

                // 分類
                if (j == 0 && i != 0)      EZF += E_mode; // ky=0: Zonal
                else if (i == 0 && j != 0) EST += E_mode; // kx=0: Streamer
                else                       Eturb += E_mode;
            }
        }

        // --- 2. 粒子束 Flux = <n * v_x> ---
        // v_x(k) = -i * ky * phi(k)
        for (int i = 0; i < NX; i++) {
            for (int j = 0; j < y_dim; j++) {
                int idx = i * y_dim + j;
                double ky = (double)j * 2*M_PI/LY;
                
                // -i * ky * (Re + iIm) = ky*Im - i*ky*Re
                double re = creal(phi_k[idx]);
                double im = cimag(phi_k[idx]);
                vx_k[idx] = (ky * im) - I * (ky * re);
            }
        }

        // 実空間へ戻す
        fftw_execute(plan_n);
        fftw_execute(plan_vx);

        // 実空間での積の平均
        double flux_sum = 0.0;
        double norm = 1.0 / (NX * NY); // 正規化
        for (int k = 0; k < NX * NY; k++) {
            flux_sum += (n_real[k] * norm) * (vx_real[k] * norm);
        }
        double flux_mean = flux_sum / (NX * NY);

        fprintf(fp_out, "%d %e %e %e %e\n", step, Eturb, EZF, EST, flux_mean);
        printf("Analyzed step %d\n", step);
    }

    fclose(fp_out);
    fftw_destroy_plan(plan_n);
    fftw_destroy_plan(plan_vx);
    fftw_free(n_k); fftw_free(phi_k); fftw_free(vx_k);
    fftw_free(n_real); fftw_free(vx_real);
    free(data_buffer);

    return 0;
}
