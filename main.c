#include "MHW.h"

int main(void) {
    // 1. 保存用ディレクトリの作成
    struct stat st = {0};
    if (stat("data", &st) == -1) {
        #ifdef _WIN32
            _mkdir("data");
        #else
            mkdir("data", 0777);
        #endif
    }

    // 2. メモリ確保 & FFTWプラン作成 (元のコードの初期化処理)
    // ※ 変数名は元のコードに合わせてください
    int complex_size = NX * (NY/2 + 1);
    cp_n   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);
    cp_phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);
    cp_vor = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);
    
    c_rhs_n   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);
    c_rhs_vor = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * complex_size);

    // プラン作成や他の配列確保が元のコードにあればここに記述
    // 例: plan_n = fftw_plan_dft_r2c_2d(...)
    
    // 3. 初期条件の設定
    init_data(); // 元のコードにある初期化関数

    // 4. パラメータの保存
    save_params();

    printf("Simulation Started. Data will be saved to 'data/' directory.\n");

    // 5. 時間発展ループ
    for (int it = 0; it <= NT; it++) {
        
        // --- データの保存 ---
        if (it % N_SAVE == 0) {
            save_data_binary(it);
            printf("Step %d / %d completed.\n", it, NT);
        }

        // --- RK4 時間積分 ---
        // 元のコードでステップを進める関数を呼ぶ
        // 例: time_step_rk4(); 
        // または、ループ内にRK4の処理が直書きされている場合はそれをここに置く
        time_step_rk4(); 
    }

    // 6. 終了処理 (メモリ解放)
    fftw_free(cp_n);
    fftw_free(cp_phi);
    fftw_free(cp_vor);
    fftw_free(c_rhs_n);
    fftw_free(c_rhs_vor);
    // fftw_destroy_plan(...) もあれば追加

    printf("Simulation Finished.\n");
    return 0;
}
