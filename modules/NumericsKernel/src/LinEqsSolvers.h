#include <vector>
#include <blaze/Math.h>
#include <iostream>
#include <cmath>
#include "KernelTypeDefs.h"


// these should be removed
template<typename MatrixType>
void fillBand(blaze::Band<MatrixType> band, GLOBAL::scalar value) {
    for(size_t i = 0; i < band.size(); i++)  band[i] = value;
}


template<typename MatrixType>
void fillBand(blaze::Band<MatrixType> band, KERNEL::vector values) {
    for(size_t i = 0; i < band.size(); i++)  band[i] = values[i];
}

template<typename MatrixType>
void checkLinEqSystemConsistency(const MatrixType& A, const KERNEL::vector& b) {
    if( blaze::isZero( A ) )
    {
        throw std::invalid_argument("Error in A is empty");
    }
    if (b.size() != A.rows())
    {
        throw std::invalid_argument("Error in B is not same side as Row of A");
    }
}


namespace LINEQSOLVERS {

    // A must be strictly diagonal dominant
    void solve_GaussSeidel( const KERNEL::dmatrix& A, KERNEL::vector& x, const KERNEL::vector& b, const GLOBAL::scalar tolerance, const unsigned int maxIter,bool printDebug = true);

    void solve_Jacobi(      const KERNEL::dmatrix &A, KERNEL::vector& x, const KERNEL::vector &b, const GLOBAL::scalar tolerance, const unsigned int maxIter,bool printDebug = true);

    bool doesJacobiConverge(const KERNEL::dmatrix &A);

    // free template class I cannot separate definition from implementation. That leads to linking error.
    template<typename MatrixType>
    void solve_BiCGSTAB(const MatrixType &A, KERNEL::vector& x, const KERNEL::vector& b, const GLOBAL::scalar tolerance, const unsigned int maxIter,bool printDebug = true) {

        const std::size_t n = A.rows();
        KERNEL::vector r0( b - A * x );
        KERNEL::vector r  = r0;
        KERNEL::vector p  = r;
        KERNEL::vector v(n, 0.0), s(n, 0.0), t(n, 0.0);

        GLOBAL::scalar rho  = blaze::dot(r0, r0);
        GLOBAL::scalar alpha = 0.0, omega = 0.0, rho1 = 0.0, beta = 0.0;

        const GLOBAL::scalar normb = std::max(blaze::norm( b ), 1e-30);
        GLOBAL::scalar normres = blaze::norm(r0);
        GLOBAL::scalar relres  = normres / normb;

        const GLOBAL::scalar norm_b = std::max(blaze::norm( b ), 1e-30);

        GLOBAL::scalar norm_res = blaze::norm(r0);
        GLOBAL::scalar rel_res  = norm_res / norm_b;

        std::size_t it = 0;
        while (rel_res > tolerance && it < maxIter)
        {
            v    = A * p;
            GLOBAL::scalar vr0 = blaze::dot(v, r0);
            if (std::fabs(vr0) < 1e-30)
            {
                std::cerr << "BiCGSTAB breakdown: v·r0 ≈ 0 → cannot compute alpha (division by zero). Stopping.\n";
                break;
            }

            alpha = rho / vr0;

            s = r - alpha * v;
            t = A * s;

            GLOBAL::scalar tt = blaze::dot(t, t);
            if (tt <= 0.0) {
                std::cerr << "BiCGSTAB breakdown: t·t == 0 → cannot compute omega (A*s = 0). Stopping iterations.\n";
                break;
            }

            omega = blaze::dot(t, s) / tt;
            if (std::fabs(omega) < 1e-30) {
                std::cerr << "BiCGSTAB breakdown: omega ≈ 0 (|omega|=" << std::fabs(omega)
                          << ") → division by omega would be unstable. Stopping iterations.\n";
                break;
            }

            x = x + alpha * p + omega * s;
            r = s - omega * t;

            rho1 = blaze::dot(r, r0);
            beta = (rho1 / rho) * (alpha / omega);
            p = r + beta * (p - omega * v);
            rho = rho1;

            norm_res = blaze::norm(r);
            rel_res  = norm_res / norm_b;
            ++it;
        }

        if (rel_res <= tolerance)
        {
            if (printDebug)
            {
                std::cout << "Bi_CG_STAB_sparse converged in " << it << " iterations\n";
            }
        }
        else
            std::cerr << "Bi_CG_STAB_sparse NOT converged (iters=" << it << ", rel res=" << rel_res << ").\n";
    }



    // use blaze's own linear equation solver for dense matrices.

    // template<typename MatrixType>
    // void solve_GaussSeidel1(const MatrixType& A, const KERNEL::vector& b, KERNEL::vector& x, const KERNEL::scalar tolerance, const unsigned int maxIter)
    // {
    //     std::cout<<"GaussSeidel1 solver started."<<std::endl;
    //
    //     auto n = A.rows();
    //     KERNEL::vector x_old = x;
    //
    //     // copy to DENSE matrix. Not good
    //     KERNEL::dmatrix DL(A);
    //
    //     // DL = lower triangular with diagonal
    //     for(size_t i = 0; i < n; ++i) {
    //         for(size_t j = i+1; j < n; ++j) {
    //             DL(i, j) = 0.0;
    //         }
    //     }
    //     std::cout << DL << std::endl;
    //
    //     // KERNEL::dmatrix B = A-DL;  // strictly upper triangular
    //
    //     // blaze::invert( DL );
    //
    //     // auto invDL = blaze::inv( DL );
    //
    //     // std::cout << invDL << std::endl;
    //     //
    //     // auto c = invDL*b;
    //     // auto G = -invDL*(A-DL);
    //
    //     std::cout << A << std::endl;
    //     std::cout << A-DL << std::endl;
    //     // std::cout << c << std::endl;
    //     // std::cout << G << std::endl;
    //
    //     // here I could free A,DL,invDL
    //     //
    //     for( int k = 0; k < maxIter; ++k )
    //     {
    //     //     // Using G and c for faster code.
    //     //     // x = G * x_old + c
    //     //     // x = invDL* (bb_-R * x_old);
    //
    //         // x = G*x_old + c;
    //
    //         if( blaze::norm( x - x_old ) < tolerance )
    //         {
    //             std::cout << "Gauss-Seidel solver converged after " << k << " iterations." << std::endl;
    //             break;
    //         }
    //         x_old = x;
    //     }
    // }

}
