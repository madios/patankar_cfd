#include <gtest/gtest.h>
#include "../../NumericsKernel/src/LinEqsSolvers.h"
#include "KERNEL_test_Structs.h"

using namespace LINEQSOLVERS;

TEST_F(NK_matrixBuilder, sparseMatrix1_BiCGSTAB)
{
    unsigned N = 2000;
    KERNEL::smatrix A(N,N, 5*N);
    KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);

    setSparseProblem_1<KERNEL::smatrix>(A, b, solution);

    solve_BiCGSTAB<KERNEL::smatrix>( A, x, b, tolerance, maxIter);

    for(int i = 0; i < solution.size(); i++)
        EXPECT_NEAR(x[i], solution[i], 1e-8);
}

TEST_F(NK_matrixBuilder, sparseMatrix2_BiCGSTAB)
{
    unsigned N = 2000;
    KERNEL::smatrix A(N,N, 5*N);
    KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);

    setSparseProblem_2<KERNEL::smatrix>(A, b, solution);

    solve_BiCGSTAB( A, x, b, tolerance, maxIter  );

    for(int i = 0; i < solution.size(); i++)
        EXPECT_NEAR(x[i], solution[i], 1e-8);
}


TEST_F(NK_matrixBuilder, denseMatrix1_BiCGSTAB)
{
    unsigned N = 600;
    KERNEL::smatrix A(N,N, 5*N);
    KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);

    setDenseProblem_1<KERNEL::smatrix>(A, b, solution);

    solve_BiCGSTAB( A, x, b, tolerance, maxIter  );

    for(int i = 0; i < solution.size(); i++)
        EXPECT_NEAR(x[i], solution[i], 1e-8);
}

TEST_F(NK_matrixBuilder, sparseMatrix1_Jacobi)
{
    unsigned N = 200;
    KERNEL::smatrix A(N,N, 5*N);
    KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);

    setSparseProblem_1<KERNEL::smatrix>(A, b, solution);

    solve_Jacobi( A, x, b, tolerance, maxIter  );

    for(int i = 0; i < solution.size(); i++)
    {
        EXPECT_NEAR(x[i], solution[i], 1e-8);
    }
}

TEST_F(NK_matrixBuilder, sparseMatrix2_Jacobi)
{
    unsigned N = 200;
    KERNEL::smatrix A(N,N, 5*N);
    KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);

    setSparseProblem_2<KERNEL::smatrix>(A, b, solution);

    solve_Jacobi( A, x, b, tolerance, maxIter  );

    for(int i = 0; i < solution.size(); i++)
        EXPECT_NEAR(x[i], solution[i], 1e-8);
}

TEST_F(NK_matrixBuilder, denseMatrix1_Jacobi)
{
    unsigned N = 40;
    KERNEL::dmatrix A(N,N, 5*N);
    KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);

    setDenseProblem_1<KERNEL::dmatrix>(A, b, solution);

    solve_Jacobi( A,x, b, tolerance, maxIter  );

    for(int i = 0; i < solution.size(); i++)
        EXPECT_NEAR(x[i], solution[i], 1e-8);
}

// TEST_F(NK_matrixBuilder, sparseMatrix1_GaussSeidel)
// {
//     unsigned N = 200;
//     KERNEL::smatrix A(N,N, 5*N);
//     KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);
//
//     setSparseProblem_1<KERNEL::smatrix>(A, b, solution);
//
//     solve_GaussSeidel( A, x, b, tolerance, maxIter  );
//
//     for(int i = 0; i < solution.size(); i++)
//         EXPECT_NEAR(x[i], solution[i], 1e-8);
// }

// TEST_F(NK_matrixBuilder, sparseMatrix2_GaussSeidel)
// {
//     unsigned N = 200;
//     KERNEL::smatrix A(N,N, 5*N);
//     KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);
//
//     setSparseProblem_2<KERNEL::smatrix>(A, b, solution);
//
//     solve_GaussSeidel( A, x, b, tolerance, maxIter  );
//
//     for(int i = 0; i < solution.size(); i++)
//         EXPECT_NEAR(x[i], solution[i], 1e-8);
// }


TEST_F(NK_matrixBuilder, denseMatrix1_GaussSeidel)
{
    unsigned N = 40;
    KERNEL::dmatrix A(N,N, 5*N);
    KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);

    setDenseProblem_1<KERNEL::dmatrix>(A, b, solution);

    solve_GaussSeidel( A, x, b, tolerance, maxIter  );

    for(int i = 0; i < solution.size(); i++)
        EXPECT_NEAR(x[i], solution[i], 1e-8);
}

