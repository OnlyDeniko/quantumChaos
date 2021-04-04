// Kulandin Denis 2021
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mkl.h"
#include <vector>
#include <iostream>
#include "mkl_lapacke.h"
#include <cassert>
#include <iomanip>
#include <complex>
#include <algorithm>

#define max(a, b) (a) < (b) ? (b): (a)

double EPS = std::numeric_limits<double>::epsilon();

lapack_int inverse(double* A, int n) {
    lapack_int* ipiv = new lapack_int[n + 1];
    lapack_int ret;

    ret = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                         n,
                         n,
                         A,
                         n,
                         ipiv);

    if (ret != 0) return ret;

    ret = LAPACKE_dgetri(LAPACK_ROW_MAJOR,
                         n,
                         A,
                         n,
                         ipiv);
    return ret;
}

void mult(double* a, double *b, double* res, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, 1.0, a, n, b, n, 0.0, res, n);
}

void print_matrix(char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda) {
    printf("\n %s\n", desc);
    for (MKL_INT i = 0; i < m; i++) {
        for (MKL_INT j = 0; j < n; j++)
            std::cout << std::fixed << std::setprecision(10) << ' ' << a[i * lda + j];
        std::cout << '\n';
    }
}

void print_matrix(char* desc, MKL_INT m, MKL_INT n, MKL_Complex16* a, MKL_INT lda) {
    printf("\n %s\n", desc);
    for (MKL_INT i = 0; i < m; i++) {
        for (MKL_INT j = 0; j < n; j++)
            std::cout << std::fixed << std::setprecision(10) << " (" << a[i * lda + j].real << "," << a[i * lda + j].imag << ")";
        std::cout << '\n';
    }
}

int main()
{
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    const MKL_INT N = 16;
    std::vector<double> C(2 * N);
    for (auto& i : C) std::cin >> i;
    fclose(stdin);

    // Ux = eSx
    double U[N * N];
    for (MKL_INT i = 0; i < N; ++i) {
        for (MKL_INT j = 0; j < N; ++j) {
            U[i * N + j] = C[i + j + 1];
        }
    }

    double inversedS[N * N], S[N * N];
    for (MKL_INT i = 0; i < N; ++i) {
        for (MKL_INT j = 0; j < N; ++j) {
            S[i * N + j] = inversedS[i * N + j] = C[i + j];
        }
    }

    auto rev = inverse(inversedS, N); // get inversed S
    assert(rev == 0);

    double doubleA[N * N];
    mult(U, inversedS, doubleA, N); // U * S^(-1) = A in double type

    // convert A to complex type
    MKL_Complex16 A[N * N];
    for (MKL_INT i = 0; i < N * N; ++i) A[i] = { doubleA[i], 0 };


    /* Solve eigenproblem Ax=ex*/
    MKL_Complex16 w[N], vl[N * N], vr[N * N];

    rev = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'V', 'V', N, A, N, w, vl,
                        N, vr, N);
    assert(rev == 0);
    // just for comfort view in output
    std::sort(w, w + N, [&](const MKL_Complex16& a, const MKL_Complex16& b) {
        return a.real > b.real;
    });

    std::cout << "Eigenvalues\n";
    for(MKL_INT i = 0;i < N; ++i){
        std::cout << std::fixed << std::setprecision(10) << " (" << w[i].real << "," << w[i].imag << ")\n";
    }
    std::cout << "log Eigenvalues\n";
    for (MKL_INT i = 0; i < N; ++i) {
        std::complex<double> cur = { w[i].real, w[i].imag };
        cur = std::log(cur);
        std::cout << std::fixed << std::setprecision(10) << " (" << cur.real() << "," << cur.imag() << ")\n";
    }
    //Solve Ax=B
    /*     (u_1^1 ... u_N^1)
           (u_1^2 ... u_N^2)
       A = (...)
           (u_1^N ... u_N^N)

       B = (C_0 ... C_N) 
    */ 
    MKL_Complex16 matrA[N * N];
    for (MKL_INT i = 0; i < N; ++i) matrA[i] = w[i];
    for (MKL_INT i = 1; i < N; ++i) {
        for (MKL_INT j = 0; j < N; ++j) {
            auto gg = std::complex<double> (matrA[(i - 1) * N + j].real, matrA[(i - 1) * N + j].imag) * 
                      std::complex<double> (matrA[j].real, matrA[j].imag);
            matrA[i * N + j] = { gg.real(), gg.imag() };
        }
    }
    MKL_Complex16 matrB[N];
    for (MKL_INT i = 0; i < N; ++i) matrB[i] = { C[i], 0 };
    MKL_INT ipiv[N];
    rev = LAPACKE_zgesv(LAPACK_ROW_MAJOR, N, 1, matrA, N, ipiv, matrB, 1);
    assert(rev == 0);
    print_matrix((char*)"Solution d_k", N, 1, matrB, 1);

    // Check C_n = sum(d_k * U_k ^ n)
    MKL_Complex16 Un[N];
    std::vector<std::complex<double>> CC(2 * N);
    for (MKL_INT i = 0; i < N; ++i) Un[i] = w[i];
    for (MKL_INT i = 0; i < 2 * N; ++i) {
        std::complex<double> ans = 0;
        for (MKL_INT j = 0; j < N; ++j) {
            std::complex<double> dj = { matrB[j].real, matrB[j].imag };
            std::complex<double> ukn = { Un[j].real, Un[j].imag };
            ans += dj * ukn;
            std::complex<double> base_w = { w[j].real, w[j].imag };
            ukn *= base_w;
            Un[j] = { ukn.real(), ukn.imag() };
        }
        CC[i] = ans;
    }
    for (MKL_INT i = 0; i < 2 * N; ++i){
        assert(fabs(C[i] - CC[i].real()) < 1e-10);
    }
    ///* Print eigenvalues */
    //print_matrix((char*)"Eigenvalues", 1, N, w, 1);
    ///* Print left eigenvectors */
    //print_matrix((char*)"Left eigenvectors", N, N, vl, N);
    ///* Print right eigenvectors */
    //print_matrix((char*)"Right eigenvectors", N, N, vr, N);
    fclose(stdout);
    return 0;
}
