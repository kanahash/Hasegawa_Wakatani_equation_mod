#include "MHW.h"

// Function to allocate 2D array (row-major)
double **alloc_2d_double(int rows, int cols) {
    double *data = (double *)malloc(rows * cols * sizeof(double));
    double **array = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        array[i] = &(data[i * cols]);
    }
    return array;
}

// Function to free 2D array
void free_2d_double(double **array) {
    free(array[0]);
    free(array);
}

// Function to allocate 2D complex array (row-major)
cplx **alloc_2d_cplx(int rows, int cols) {
    cplx *data = (cplx *)fftw_malloc(rows * cols * sizeof(cplx));
    cplx **array = (cplx **)malloc(rows * sizeof(cplx *));
    for (int i = 0; i < rows; i++) {
        array[i] = &(data[i * cols]);
    }
    return array;
}

// Function to free 2D complex array (uses fftw_free)
void free_2d_cplx(cplx **array) {
    fftw_free(array[0]);
    free(array);
}
