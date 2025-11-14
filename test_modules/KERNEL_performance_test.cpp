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

    const std::vector<int> sizes = {1000,2000,4000,8000,16000,32000,64000};
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
        EXPECT_NEAR(x[2], solution[2], 1e-8);
    }
}