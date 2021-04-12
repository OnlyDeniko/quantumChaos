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

lapack_int inverse(std::vector<double>& A, int n) {
    std::vector<lapack_int> ipiv(n + 1);
    lapack_int ret;

    ret = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                         n,
                         n,
                         A.data(),
                         n,
                         ipiv.data());

    if (ret != 0) return ret;

    ret = LAPACKE_dgetri(LAPACK_ROW_MAJOR,
                         n,
                         A.data(),
                         n,
                         ipiv.data());
    return ret;
}

void mult(std::vector<double>& a, std::vector<double>& b, std::vector<double>& res, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, 1.0, a.data(), n, b.data(), n, 0.0, res.data(), n);
}

void print_matrix(char* desc, MKL_INT m, MKL_INT n, const std::vector<double>& a, MKL_INT lda) {
    printf("\n %s\n", desc);
    for (MKL_INT i = 0; i < m; i++) {
        for (MKL_INT j = 0; j < n; j++)
            std::cout << std::fixed << std::setprecision(10) << ' ' << a[i * lda + j];
        std::cout << '\n';
    }
}

void print_matrix(char* desc, MKL_INT m, MKL_INT n, const std::vector<MKL_Complex16>& a, MKL_INT lda) {
    //printf("\n %s\n", desc);
    std::cout << '\n';
    for (MKL_INT i = 0; i < m; i++) {
        for (MKL_INT j = 0; j < n; j++)
            std::cout << std::fixed << std::setprecision(10) << a[i * lda + j].real << " " << a[i * lda + j].imag;
        if (i != m - 1) std::cout << '\n';
    }
}

void solve(const std::vector<double>& C,
           const MKL_INT N,
           std::vector<MKL_Complex16>& w,
           std::vector<MKL_Complex16>& matrB) {
    // Ux = eSx
    std::vector<double> U(N * N);
    for (MKL_INT i = 0; i < N; ++i) {
        for (MKL_INT j = 0; j < N; ++j) {
            U[i * N + j] = C[i + j + 1];
        }
    }

    std::vector<double> inversedS(N * N), S(N * N);
    for (MKL_INT i = 0; i < N; ++i) {
        for (MKL_INT j = 0; j < N; ++j) {
            S[i * N + j] = inversedS[i * N + j] = C[i + j];
        }
    }

    auto rev = inverse(inversedS, N); // get inversed S
    if (rev > 0) {
        std::cout << "Error occured after inverse\n";
        exit(1);
    }

    std::vector<double> doubleA(N * N);
    mult(U, inversedS, doubleA, N); // U * S^(-1) = A in double type

    // convert A to complex type
    std::vector<MKL_Complex16> A(N * N);
    for (MKL_INT i = 0; i < N * N; ++i) A[i] = { doubleA[i], 0 };


    /* Solve eigenproblem Ax=ex*/
    std::vector<MKL_Complex16> vl(N * N), vr(N * N);

    rev = LAPACKE_zgeev(LAPACK_ROW_MAJOR, 'V', 'V', N, A.data(), N, w.data(), vl.data(),
        N, vr.data(), N);
    if (rev > 0) {
        std::cout << "Error occured after LAPACKE_zgeev\n";
        exit(1);
    }
    
    /*std::cout << "log Eigenvalues\n";
    for (MKL_INT i = 0; i < N; ++i) {
        std::complex<double> cur = { w[i].real, w[i].imag };
        cur = std::log(cur);
        std::cout << std::fixed << std::setprecision(10) << " (" << cur.real() << "," << cur.imag() << ")\n";
    }*/

    //Solve Ax=B
    /*     (u_1^1 ... u_N^1)
           (u_1^2 ... u_N^2)
       A = (...)
           (u_1^N ... u_N^N)

       B = (C_0 ... C_N)
    */
    std::vector<MKL_Complex16> matrA(N * N);
    for (MKL_INT i = 0; i < N; ++i) matrA[i] = w[i];
    for (MKL_INT i = 1; i < N; ++i) {
        for (MKL_INT j = 0; j < N; ++j) {
            auto gg = std::complex<double>(matrA[(i - 1) * N + j].real, matrA[(i - 1) * N + j].imag) *
                std::complex<double>(matrA[j].real, matrA[j].imag);
            matrA[i * N + j] = { gg.real(), gg.imag() };
        }
    }
    for (MKL_INT i = 0; i < N; ++i) matrB[i] = { C[i], 0 };
    std::vector<MKL_INT> ipiv(N);
    rev = LAPACKE_zgesv(LAPACK_ROW_MAJOR, N, 1, matrA.data(), N, ipiv.data(), matrB.data(), 1);
    if (rev > 0) {
        std::cout << "Error occured after LAPACKE_zgesv\n";
        exit(1);
    }

    // Check C_n = sum(d_k * U_k ^ n)
    std::vector<MKL_Complex16> Un(N);
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
    for (MKL_INT i = 0; i < 2 * N; ++i) {
        // assert(fabs(C[i] - CC[i].real()) < 1e-10);
    }
    ///* Print eigenvalues */
    //print_matrix((char*)"Eigenvalues", 1, N, w, 1);
    ///* Print left eigenvectors */
    //print_matrix((char*)"Left eigenvectors", N, N, vl, N);
    ///* Print right eigenvectors */
    //print_matrix((char*)"Right eigenvectors", N, N, vr, N);
}

int main()
{
    freopen("input.txt", "r", stdin);
    freopen("output5.txt", "w", stdout);
    std::vector<double> C(32);
    for (auto& i : C) std::cin >> i;
    fclose(stdin);

    for (int N = 2; N <= 16; N += 1){
        std::vector<double> tmp(C.begin(), C.begin() + 2 * N);
        std::cout << N;
        std::vector<MKL_Complex16> w(N);
        std::vector<MKL_Complex16> matrB(N);
        solve(tmp, N, w, matrB);
        // just for comfort view in output
        std::sort(w.begin(), w.end(), [&](const MKL_Complex16& a, const MKL_Complex16& b) {
            std::complex<double> aa(a.real, a.imag);
            std::complex<double> bb(b.real, b.imag);
            return std::abs(aa) > std::abs(bb);
        });
        std::sort(matrB.begin(), matrB.end(), [&](const MKL_Complex16& a, const MKL_Complex16& b) {
            std::complex<double> aa(a.real, a.imag);
            std::complex<double> bb(b.real, b.imag);
            return std::abs(aa) > std::abs(bb);
        });
        print_matrix((char*)"Eigenvalues", N, 1, w, 1);
        print_matrix((char*)"Solution d_k", N, 1, matrB, 1);
        std::cout << '\n';
    }
    
    fclose(stdout);
    return 0;
}
