

struct NK_matrixBuilder : public::testing::Test {

    KERNEL::scalar tolerance = 1e-15;
    unsigned int maxIter = 100000;

    //    ( 2 1 0 0 0 0 )    ( x_0 )       (1)
    //    ( 0 2 1 0 0 0 )    ( x_1 )       (2)
    //    ( 0 0 2 1 0 0 )    ( x_2 )       (1)
    //    ( 0 0 0 2 1 0 )  * ( x_3 )    =  (2)
    //    ( 0 0 0 0 2 1 )    ( x_4 )       (1)
    //    ( 0 0 0 0 0 2 )    ( x_5 )       (2)
    //
    // solution: x = (0, 1, 0, 1, 0, 1)ˆT
    template<typename MatrixType>
    void setSparseProblem_1(MatrixType& A, KERNEL::vector& b, KERNEL::vector& solution ) {

        fillBand<MatrixType>( blaze::band(A,0), 2.0 );
        fillBand<MatrixType>( blaze::band(A,1), 1.0 );

        std::fill(b.begin(), b.end(), 1.0);
        std::fill(solution.begin(), solution.end(), 0.0);
        for (int i = 0; i < b.size(); i++)
        {
            if (i % 2 != 0) {
                b[i]++;
                solution[i]++;
            }
        }
    }


    //    ( -2 0 0 0 1 0 0 0 0 0 0 0 ...)    ( x_0 )       (3)
    //    ( 0 -2 0 0 0 1 0 0 0 0 0 0 ...)    ( x_1 )       (2)
    //    ( 0 0 -2 0 0 0 1 0 0 0 0 0 ...)    ( x_2 )       (1)
    //    ( 0 0 0 -2 0 0 0 1 0 0 0 0 ...)  * ( x_3 )    =  (0)
    //    ( 1 0 0 0 0 -2 0 0 0 1 0 0 ...)    ( x_4 )       (0)
    //    ( 0 1 0 0 0 0 -2 0 0 0 1 0 ...)    ( x_5 )       (0)
    //              ......
    //    ( ... 0 0 0 1 0 0 0 -2 0 0 0 1)    ( x_N-5 )      (0)
    //    ( ... 0 0 0 0 1 0 0 0 -2 0 0 0)    ( x_N-4 )      (-N-1)
    //    ( ... 0 0 0 0 0 1 0 0 0 -2 0 0)    ( x_N-3 )      (-N-2)
    //    ( ... 0 0 0 0 0 0 1 0 0 0 -2 0)    ( x_N-2 )      (-N-3)
    //    ( ... 0 0 0 0 0 0 0 1 0 0 0 -2)    ( x_N-1 )      (-N-4)
    //    solution: x = (1, 2, 3, 4, ... , N)ˆT
    template<typename MatrixType>
    void setSparseProblem_2(MatrixType& A, KERNEL::vector& b, KERNEL::vector& solution )
    {
        //only for problems bigger then 7
        fillBand<MatrixType>( blaze::band(A,0), -2.0 );
        fillBand<MatrixType>( blaze::band(A,4), 1.0 );
        fillBand<MatrixType>( blaze::band(A,-4), 1.0 );

        auto N = b.size();
        b[0] = 3.0;
        b[1] = 2.0;
        b[2] = 1.0;
        b[N-4] = -static_cast<double>(N)-1.0;
        b[N-3] = -static_cast<double>(N)-2.0;
        b[N-2] = -static_cast<double>(N)-3.0;
        b[N-1] = -static_cast<double>(N)-4.0;

        std::iota(solution.begin(), solution.end(), 1.0);
    };

    //    Q = 0.5*N*(N+1)
    //
    //    ( 2 1 1 1 1 ... )    ( x_0 )       ( 1 + Q )
    //    ( 1 2 1 1 1 ... )    ( x_1 )       ( 2 + Q )
    //    ( 1 1 2 1 1 ... )    ( x_2 )   =   ( 3 + Q )
    //    ( 1 1 1 2 1 ... )    ( x_3 )       ( 4 + Q )
    //           ...              ...           ...
    //    ( ... 1 1 1 1 2 )    ( x_(N-1) )   ( N + Q )
    //    solution: x = (1, 2, 3, 4, ... , N)ˆT
    template<typename MatrixType>
    void setDenseProblem_1(MatrixType& A, KERNEL::vector& b, KERNEL::vector& solution )
    {
        auto N = A.rows();
        for (size_t  i = 0; i<N; ++i) {
            for (size_t j = 0; j<N; ++j) {
                A(i,j) = 1.0;
            }
        }
        fillBand<MatrixType>( blaze::band(A,0), 2.0 );

        auto Q = 0.5*N*(N+1);
        std::iota(b.begin(), b.end(), 1+Q);
        std::iota(solution.begin(), solution.end(), 1.0);
    }

};
