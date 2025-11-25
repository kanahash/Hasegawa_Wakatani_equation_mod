#include "MHW.h"

// --- Memory Allocation/Deallocation Implementations ---

/**
 * @brief 2次元 double 配列を連続メモリ領域として確保する。
 * 行ポインタ配列 (array) と データ領域 (data) をセットで確保します。
 */
double **alloc_2d_double(int rows, int cols) {
    // データの実体を1次元配列として連続確保
    double *data = (double *)malloc(rows * cols * sizeof(double));
    // 行の先頭アドレスを格納するポインタ配列
    double **array = (double **)malloc(rows * sizeof(double *));
    
    if (data == NULL || array == NULL) {
        if (data != NULL) free(data);
        if (array != NULL) free(array);
        fprintf(stderr, "Memory allocation failed for double array (%dx%d).\n", rows, cols);
        return NULL;
    }

    // 各行のポインタをセット
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
        // array[0] はデータ領域(data)の先頭を指しているため、これを解放すれば実体は消える
        if (array[0] != NULL) free(array[0]);
        // ポインタ配列自体を解放
        free(array);
    }
}

/**
 * @brief 2次元 cplx (FFTW対応) 配列を連続メモリ領域として確保する。
 * 重要: FFTWの関数に渡すため、fftw_malloc を使用し、連続領域であることを保証する。
 */
cplx **alloc_2d_cplx(int rows, int cols) {
    // FFTW専用のメモリ確保 (SIMD命令などで高速化されるため推奨)
    cplx *data = (cplx *)fftw_malloc(rows * cols * sizeof(cplx));
    cplx **array = (cplx **)malloc(rows * sizeof(cplx *));

    if (data == NULL || array == NULL) {
        if (data != NULL) fftw_free(data);
        if (array != NULL) free(array);
        fprintf(stderr, "Memory allocation failed for cplx array (%dx%d).\n", rows, cols);
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
        // 実体は fftw_malloc で確保したので fftw_free で解放
        if (array[0] != NULL) fftw_free(array[0]);
        free(array);
    }
}
