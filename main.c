#include "MHW.h"

// --- シミュレーション設定 ---
#define NX 256
#define NY 256
#define LX 50.0
#define LY 50.0
#define DT 0.005
#define NT 4000
#define KAP 1.0
#define ALPH 1.0
#define MU 0.01
#define NU 0.01

// 修正箇所: 引数を (void) に変更して警告を回避
int main(void) {
    // 1. 初期条件用メモリ確保 (MHW.h の関数を使用)
    // 行: nx, 列: ny
    double **phi_init = alloc_2d_double(NX, NY);
    double **n_init = alloc_2d_double(NX, NY);

    // 2. 初期条件の設定
    // ここではゼロ初期化（MHW内部で微小ノイズが加わる想定）
    for(int i = 0; i < NX; i++){
        for(int j = 0; j < NY; j++){
            phi_init[i][j] = 0.0;
            n_init[i][j] = 0.0;
        }
    }

    // 3. データ保存ディレクトリ
    const char *data_dir = "data";
    
    // ディレクトリ作成
    struct stat st = {0};
    if (stat(data_dir, &st) == -1) {
        // Windows/Linux 両対応のため #ifdef を入れるのが一般的ですが、
        // 以前のコードに合わせて単純な mkdir にしています
        #ifdef _WIN32
            _mkdir(data_dir);
        #else
            mkdir(data_dir, 0777);
        #endif
    }
    
    printf("Starting MHW simulation...\n");
    printf("NX=%d, NY=%d, NT=%d\n", NX, NY, NT);

    // 4. メイン関数の呼び出し
    MHW(NX, NY, LX, LY, NT, DT, KAP, ALPH, MU, NU, phi_init, n_init, 100, data_dir);

    // 5. 終了処理
    free_2d_double(phi_init);
    free_2d_double(n_init);

    printf("Simulation completed.\n");

    return 0;
}
