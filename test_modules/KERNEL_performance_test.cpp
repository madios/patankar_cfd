#include <gtest/gtest.h>
#include "../../NumericsKernel/src/LinEqsSolvers.h"
#include "KERNEL_test_Structs.h"

using namespace LINEQSOLVERS;

class timer {
    private:
    std::chrono::high_resolution_clock::time_point start_time;
    public:
    void start(){
        start_time = std::chrono::high_resolution_clock::now();
    }

    long long stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }
};

TEST_F(NK_matrixBuilder, performance_BiCGSTAB_sparseMatrix1)
{

    auto timer1 = timer();
    auto timer2 = timer();
    auto timer3 = timer();

    const std::vector<int> sizes = {125,250,500,1000,2000,4000,8000,16000,32000,64000};
    for (int i = 0; i<sizes.size(); i++) {
        auto N = sizes[i];
        timer1.start();
        KERNEL::smatrix A(N,N, 5*N);
        KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);

        timer2.start();
        setSparseProblem_1<KERNEL::smatrix>(A, b, solution);
        //setSparseProblem_2<KERNEL::smatrix>(A, b, solution);
        //setDenseProblem_1<KERNEL::smatrix>(A, b, solution);

        timer3.start();
        solve_BiCGSTAB<KERNEL::smatrix>( A, x, b, tolerance, maxIter);
        auto T3 = timer3.stop();
        auto T2 = timer2.stop();
        auto T1 = timer1.stop();

        std::cout << N << ", \t" << T1 << " ms, \t" << T2 << " ms, \t" << T3 << " ms " << std::endl;
        EXPECT_NEAR(x[2], solution[2], TestTolerance);
    }
}

TEST_F(NK_matrixBuilder, performance_Jacobi_sparseMatrix1)
{

    auto timer1 = timer();
    auto timer2 = timer();
    auto timer3 = timer();

    const std::vector<int> sizes = {1000,2000,4000,8000,16000,32000,64000};
    for (int i = 0; i<sizes.size(); i++) {
        auto N = sizes[i];
        timer1.start();
        KERNEL::smatrix A(N,N, 5*N);
        KERNEL::vector b(N,0.0), x(N, 0.0), solution(N, 0.0);
        timer2.start();
        setSparseProblem_1<KERNEL::smatrix>(A, b, solution);

        timer3.start();
        bool checkConverge = doesJacobiConverge(A);
        solve_Jacobi( A, x, b, AlgoTolerance, maxIter);
        auto T3 = timer3.stop();
        auto T2 = timer2.stop();
        auto T1 = timer1.stop();

        std::cout << N << ", \t" << T1 << " ms, \t" << T2 << " ms, \t" << T3 << " ms " << std::endl;
        EXPECT_NEAR(x[2], solution[2], 1e-8);
    }
}

TEST_F(NK_matrixBuilder, performance_testing_all)
{
    auto timer3 = timer();

    const std::vector<int> sizes = {4000,8000,16000,32000,64000, 12800,25600};

    // Column layout
    const int nameW = 12;
    const int colW  = 12;

    // Header
    std::cout << std::left << std::setw(nameW) << "Methods";
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << std::right << std::setw(colW) << sizes[i];
    }
    std::cout << std::endl;

    auto printRow = [&](const std::string& methodName, const std::vector<long long>& timesMs)
    {
        std::cout << std::left << std::setw(nameW) << (methodName);
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (i < timesMs.size()) {
                std::ostringstream cell;
                cell << timesMs[i] << " ms";
                std::cout << std::right << std::setw(colW) << cell.str();
            } else {
                // Print empty cells for methods that are not run for all sizes
                std::cout << std::right << std::setw(colW) << "-";
            }
        }
        std::cout << std::endl;
    };

    // BiCGSTAB
    {
        std::vector<long long> times;
        times.resize(sizes.size());
        for (size_t i = 0; i < sizes.size(); ++i) {
            auto N = sizes[i];
            KERNEL::smatrix A(N, N, 5 * N);
            KERNEL::vector b(N, 0.0), x(N, 0.0), solution(N, 0.0);
            setSparseProblem_1<KERNEL::smatrix>(A, b, solution);

            timer3.start();
            solve_BiCGSTAB<KERNEL::smatrix>(A, x, b, TestTolerance, maxIter, false);
            auto solveTime = timer3.stop();
            times[i] = solveTime;
        }
        printRow("BiCGSTAB", times);
    }

    // Blaze solver
    {
        std::vector<long long> times;
        const size_t nRun = 3;
        times.resize(nRun);
        for (size_t i = 0; i < nRun; ++i) {
            auto N = sizes[i];
            KERNEL::smatrix A(N, N, 5 * N);
            KERNEL::vector b(N, 0.0), x(N, 0.0), solution(N, 0.0);
            setSparseProblem_1<KERNEL::smatrix>(A, b, solution);

            KERNEL::dmatrix A_dense( N, N, 0.0 );   // DynamicMatrix<double>
            A_dense = A;
            timer3.start();
            blaze::solve(A_dense, x, b);
            auto solveTime = timer3.stop();
            times[i] = solveTime;


        }
        printRow("Blaze solver", times);
    }

    // Jacobi
    {
        std::vector<long long> times;
        const size_t nRun = 3;
        times.resize(nRun);
        for (size_t i = 0; i < nRun; ++i) {
            auto N = sizes[i];
            KERNEL::smatrix A(N, N, 5 * N);
            KERNEL::vector b(N, 0.0), x(N, 0.0), solution(N, 0.0);
            setSparseProblem_1<KERNEL::smatrix>(A, b, solution);

            timer3.start();
            solve_Jacobi(A, x, b, TestTolerance, maxIter, false);
            auto solveTime = timer3.stop();
            times[i] = solveTime;
        }
        printRow("Jacobi", times);
    }

    // GaussSeidel
    {
        std::vector<long long> times;
        const size_t nRun = 1;
        times.resize(nRun);
        for (size_t i = 0; i < nRun; ++i) {
            auto N = sizes[i];
            KERNEL::smatrix A(N, N, 5 * N);
            KERNEL::vector b(N, 0.0), x(N, 0.0), solution(N, 0.0);
            setSparseProblem_1<KERNEL::smatrix>(A, b, solution);

            timer3.start();
            solve_GaussSeidel(A, x, b, TestTolerance, maxIter, false);
            auto solveTime = timer3.stop();
            times[i] = solveTime;
        }
        printRow("GaussSeidel", times);
    }
}