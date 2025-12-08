#include <gtest/gtest.h>
#include "KERNEL.h"
#include <numeric>
#include "../../NumericsKernel/src/LinEqsSolvers.h"
#include "blaze/Blaze.h"

template< typename T >
void printBlaze( const T& expr,
                 const std::string& name = "",
                 int width = 10,
                 int precision = 4 )
{
    std::ostream& os = std::cout;
    // Fjern evt. proxy / reference via ~ (Blaze expression)
    const auto& x = ~expr;

    using BlazeType = blaze::RemoveReference_t< decltype( x ) >;

    os.setf( std::ios::fixed );
    os << std::setprecision( precision );

    if( !name.empty() ) {
        os << name;
    }

    // ----- Matrix -----
    if constexpr( blaze::IsMatrix_v<BlazeType> )
    {
        const size_t m = x.rows();
        const size_t n = x.columns();

        if( !name.empty() ) {
            os << " (" << m << "x" << n << ")";
        }
        os << ":\n";

        for( size_t i = 0; i < m; ++i )
        {
            os << "(";
            for( size_t j = 0; j < n; ++j )
            {
                os << std::setw( width ) << x(i,j);
            }
            os << " )\n";
        }
        os << std::endl;
    }

    // ----- Vector -----
    else if constexpr( blaze::IsVector_v<BlazeType> )
    {
        const size_t n = x.size();

        if( !name.empty() ) {
            os << " (" << n << ")";
        }
        os << ":\n";

        // Row vector
        if constexpr( blaze::IsRowVector_v<BlazeType> )
        {
            os << "(";
            for( size_t i = 0; i < n; ++i )
            {
                os << std::setw( width ) << x[i];
            }
            os << " )\n\n";
        }
        // Column vector
        else
        {
            for( size_t i = 0; i < n; ++i )
            {
                os << "(" << std::setw( width ) << x[i] << " )\n";
            }
            os << std::endl;
        }
    }

    // Hvis det hverken er matrix eller vektor → compile-time fejl
    else {
        static_assert( blaze::IsMatrix_v<BlazeType> || blaze::IsVector_v<BlazeType>,
                       "printBlaze() expects a Blaze matrix or vector type" );
    }
}

template< typename V1, typename V2 >
void printBlazeVectorsSideBySide( const V1& a_in,
                                  const V2& b_in,
                                  const std::string& nameA = "a",
                                  const std::string& nameB = "b",
                                  std::ostream& os = std::cout,
                                  int width = 12,
                                  int precision = 6 )
{
    const auto& a = ~a_in;
    const auto& b = ~b_in;

    using TA = blaze::RemoveReference_t< decltype(a) >;
    using TB = blaze::RemoveReference_t< decltype(b) >;

    static_assert( blaze::IsVector_v<TA>, "First argument must be a Blaze vector" );
    static_assert( blaze::IsVector_v<TB>, "Second argument must be a Blaze vector" );

    if( a.size() != b.size() ) {
        os << "ERROR: Vectors have different sizes ("
           << a.size() << " vs " << b.size() << ")\n";
        return;
    }

    const size_t n = a.size();

    // Format
    os.setf( std::ios::fixed );
    os << std::setprecision( precision );

    // Header
    os << "Index" << std::setw(width) << nameA << std::setw(width) << nameB << "\n";
    os << std::string(5 + 2*width, '-') << "\n";

    // Print element by element
    for( size_t i = 0; i < n; ++i )
    {
        os << std::setw(5) << i
           << std::setw(width) << a[i]
           << std::setw(width) << b[i]
           << "\n";
    }

    os << std::endl;
}


struct kernelInterface : public ::testing::Test {
};

TEST_F(kernelInterface, interfaceTest) {

    KERNEL::ObjectRegistry objReg;
    auto vecHandle = objReg.newVector(5, 3);
    auto matHandleDense = objReg.newMatrix(5, 3, false);
    auto matHandleSparse = objReg.newMatrix(5, 3, true);

    // I have to close the registry, before I am allowed to access its objects:
    EXPECT_THROW(objReg.getVectorRef(vecHandle),std::runtime_error);

    // now it is closed.
    objReg.closeRegistry();

    // These are references. They may go out of scope, the objects survive in the reg.
    auto u = objReg.getVectorRef(vecHandle);
    auto A  = objReg.getDenseMatrixRef(matHandleDense);
    auto B  = objReg.getSparseMatrixRef(matHandleSparse);

    // It is forbidden to add new objects in the registry, for keeping the existing
    // references valid. The following line will fail:
    EXPECT_THROW(objReg.newMatrix(5, 3, false),std::runtime_error);

}

// constructing a square domain with
struct FVM_laplaceTests : public ::testing::Test {
    static constexpr double tolerance = 1e-6;
    unsigned int nx, ny, nbCells;
    double lenx, leny, cellSpacing, faceArea;
    KERNEL::MatrixHandle AHandle;
    KERNEL::VectorHandle uHandle;
    KERNEL::VectorHandle bHandle;

    // these are to be fetched from a future MESH class
    std::vector<unsigned int> cellIndices_North;
    std::vector<unsigned int> cellIndices_South;
    std::vector<unsigned int> cellIndices_East;
    std::vector<unsigned int> cellIndices_West;

    KERNEL::ObjectRegistry setUp(KERNEL::scalar length, unsigned int nbX)
    {
        if (nbX%2 == 0)
            throw std::runtime_error("ERROR: nx must be uneven");

        nx = nbX;
        ny = nbX;
        nbCells = nx*ny;
        leny = length;
        lenx = length;
        KERNEL::ObjectRegistry objReg;
        AHandle = objReg.newMatrix(nbCells, nbCells, true);
        uHandle = objReg.newVector(nbCells);
        bHandle = objReg.newVector(nbCells);
        objReg.closeRegistry();

        cellSpacing = lenx/static_cast<double>(nx);
        faceArea = cellSpacing * cellSpacing;

        cellIndices_North.resize(nx);
        cellIndices_South.resize(nx);
        cellIndices_East.resize(ny);
        cellIndices_West.resize(ny);

        for (unsigned int i = 0; i < nx; i++) {
            cellIndices_North[i] = i;
            cellIndices_South[i] = i+nx*(ny-1);
            cellIndices_East[i] = i*nx+nx-1;
            cellIndices_West[i] = i*nx;
        }
        return objReg;
    }

};


TEST_F(FVM_laplaceTests, FVM_testtest) {

    // nb of cells along one side
    auto nbX = 81;

    // setup() fills the registry with a matrix and a vector,
    // the handles are members of the test struct
    auto objReg = setUp(1.0, nbX);

    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);
    // auto A = objReg.getDenseMatrixRef(AHandle);
    auto A = objReg.getSparseMatrixRef(AHandle);

    // building matrix
    auto diagonal = blaze::band(A,0);
    for (unsigned int i = 0; i < diagonal.size(); i++) {
        diagonal[i] = 4.0;
    }

    auto aw = blaze::band(A,-1);
    for (unsigned int i = 0; i < aw.size(); i++) {
        aw[i] = 1.0;
    }

    auto ae = blaze::band(A,1);
    for (unsigned int i = 0; i < ae.size(); i++) {
        ae[i] = 2.0;
    }

    auto correctResult = *KERNEL::newTempVector(b.size());
    std::iota(correctResult.begin(),correctResult.end(),1);

    b = A*correctResult;

    // an educated guess for u:
    std::ranges::fill(u, 2.0);

    KERNEL::solve(A, u, b, 1e-10, 1000, KERNEL::BiCGSTAB);

    for (unsigned int i = 0; i < correctResult.size(); i++) {
        EXPECT_NEAR(u[i], correctResult[i], 1e-4);
    }
}



// 2D Laplace Equation with local BCs
//
// Problem Setup:
// - Domain: 0 ≤ x ≤ nx, 0 ≤ y ≤ ny (rectangular)
// - ODE: ∇²φ = 0
//
// Boundary Conditions:
// - Dirichlet: φ(x,0) = 0, φ(0,y) = 0, φ(x,1) = x*ny,  φ(1,y) = y*nx
//
// Analytical Solution:
// φ(x,y) = y*x
TEST_F(FVM_laplaceTests, FVM_localDerichletBCs) {

    auto objReg = setUp(1.0, 11);

    auto A = objReg.getSparseMatrixRef(AHandle);
    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);

    KERNEL::vector ae(nbCells, 0.0),aw(nbCells, 0.0),an(nbCells, 0.0),as(nbCells, 0.0),ap(nbCells, 0.0),sp(nbCells, 0.0),su(nbCells, 0.0);

    for (unsigned int i=0; i < nbCells; i++)
    {
            ae[i] = aw[i] = an[i] = as[i] = faceArea / cellSpacing;
            ap[i] = -4 * faceArea / cellSpacing;
            sp[i] = 0;
    }

    // EAST
    for (unsigned int i = 0; i < ny; i++) {
        auto xpos = lenx;
        auto ypos = leny - (0.5+i)*cellSpacing;
        auto j = nx-1 +i*nx;
        ae[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * ypos * xpos;
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    // NORTH
    for (unsigned int i = 0; i < nx; i++) {
        auto xpos = (0.5*cellSpacing + i*cellSpacing);
        auto ypos = leny;
        an[i] = 0.0;
        b[i] -= 2*faceArea/cellSpacing * xpos * ypos;
        sp[i] -= 2*faceArea/cellSpacing;
        ap[i] -= faceArea/cellSpacing;
    }

    // WEST
    for (unsigned int i = 0; i < ny; i++) {
        auto xpos = 0.0;
        auto ypos = (leny - 0.5*cellSpacing - i * cellSpacing);
        auto j = i*nx;
        aw[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * xpos*ypos;
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    // SOUTH
    for (unsigned int i = 0; i < nx; i++) {
        auto xpos = (0.5*cellSpacing + i*cellSpacing);
        auto ypos = 0.0;
        auto j = (ny-1)*nx + i;
        as[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * xpos*ypos;
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    for (unsigned int i = 0; i < ap.size() -1; i++) {
        blaze::band(A,1)[i] = ae[i];
    }
    for (unsigned int i = 0; i < ap.size()-1; i++) {
        blaze::band(A,-1)[i] = aw[i+1];
    }
    for (unsigned int i = 0; i < ap.size()-nx; i++) {
        blaze::band(A,nx)[i] = as[i];
    }
    int mnx = -1*nx;        // I don't see the necessity for that
    for (unsigned int i = 0; i < ap.size()-nx; i++) {
        blaze::band(A,mnx)[i] = an[i+nx];
    }
    KERNEL::vector ap1(ap.size(),0);
    for (unsigned int i = 0; i < ap.size(); i++)
    {
        ap1[i] = -(ae[i] + aw[i] + as[i] + an[i] - sp[i]);
    }
    for (unsigned int i = 0; i < ap.size(); i++)
    {
        blaze::band(A,0)[i] = ap1[i];
    }

    KERNEL::solve(A, u, b, 1e-10, 1000, KERNEL::BiCGSTAB);

    KERNEL::vector solution( nx, 0.0 );
    for (unsigned int i=0; i < nx; i++) {
        auto x = 0.5*lenx;
        auto y = leny - ( 0.5 + i )*cellSpacing;
        solution[i] = x*y;
    }

    for(int i = 0; i < solution.size(); i++)
    {
        auto j = nx/2 + nx*i;
        EXPECT_NEAR(u[j], solution[i],tolerance);
    }
}
void buildMatrixWithBandsSpeed( KERNEL::smatrix &A,
                           const KERNEL::vector &ae,
                           const KERNEL::vector &aw,
                           const KERNEL::vector &as,
                           const KERNEL::vector &an,
                           const KERNEL::vector &sp,
                           const KERNEL::vector &ap,
                           std::size_t           nx )
{
    using std::size_t;

    const size_t N = ap.size();

    // Max ca. 5 ikke-nul per række (W, E, N, S, diag)
    const size_t nnzPerRow = 5;
    A.reserve( N * nnzPerRow );
    for( size_t i = 0; i < N; ++i ) {
        A.reserve( i, nnzPerRow );
    }

    // --- East AE
    auto b1 = blaze::band( A, 1 );
    for( size_t i = 0; i < N-1; ++i ) {
        b1[i] = ae[i];
    }

    // --- West AW
    auto bm1 = blaze::band( A, -1 );
    for( size_t i = 0; i < N-1; ++i ) {
        bm1[i] = aw[i+1];
    }

    // --- "Syd" AS
    auto bnx = blaze::band( A, static_cast<std::ptrdiff_t>( nx ) );
    for( size_t i = 0; i < N - nx; ++i ) {
        bnx[i] = as[i];
    }
    // --- "Nord" AN
    auto bmnx = blaze::band( A, -static_cast<std::ptrdiff_t>( nx ) );
    for( size_t i = 0; i < N - nx; ++i ) {
        bmnx[i] = an[i+nx];
    }

    // --- Diagonal AP
    auto b0 = blaze::band( A, 0 );
    for( size_t i = 0; i < N; ++i ) {
        b0[i] = -( ae[i] + aw[i] + as[i] + an[i] - sp[i] );
    }
}

TEST_F(FVM_laplaceTests, spacVarDerichletBCsSpeed)
{

    //auto objReg = setUp(1, 161);
    auto objReg = setUp(1, 161);

    auto A = objReg.getSparseMatrixRef(AHandle);
    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);
    KERNEL::smatrix A1(A.rows(),A.columns());


    KERNEL::vector ae(nbCells, 0.0),aw(nbCells, 0.0),an(nbCells, 0.0),as(nbCells, 0.0),ap(nbCells, 0.0),sp(nbCells, 0.0),su(nbCells, 0.0);

    for (unsigned int i=0; i < nbCells; i++)
    {
        ae[i] = aw[i] = an[i] = as[i] = faceArea / cellSpacing;
        ap[i] = -4 * faceArea / cellSpacing;
        sp[i] = 0;
    }

    // EAST
    for (unsigned int i = 0; i < ny; i++) {
        auto xpos = lenx;
        auto ypos = leny - (0.5+i)*cellSpacing;
        auto j = nx-1 +i*nx;
        ae[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * (xpos*xpos-ypos*ypos);
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    // NORTH
    for (unsigned int i = 0; i < nx; i++) {
        auto xpos = (0.5*cellSpacing + i*cellSpacing);
        auto ypos = leny;
        an[i] = 0.0;
        b[i] -= 2*faceArea/cellSpacing * (xpos*xpos-ypos*ypos);
        sp[i] -= 2*faceArea/cellSpacing;
        ap[i] -= faceArea/cellSpacing;
    }

    // WEST
    for (unsigned int i = 0; i < ny; i++) {
        auto xpos = 0.0;
        auto ypos = (leny - 0.5*cellSpacing - i * cellSpacing);
        auto j = i*nx;
        aw[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * (xpos*xpos-ypos*ypos);
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    // SOUTH
    for (unsigned int i = 0; i < nx; i++) {
        auto xpos = (0.5*cellSpacing + i*cellSpacing);
        auto ypos = 0.0;
        auto j = (ny-1)*nx + i;
        as[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * (xpos*xpos-ypos*ypos);
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    buildMatrixWithBandsSpeed( A, ae, aw, as, an,sp, ap, nx);
    KERNEL::solve(A, u, b, 1e-15, 2000, KERNEL::BiCGSTAB);

    // theoretical solution, vertical mid-line at x = lenx/2
    KERNEL::vector solution( nx, 0.0 );
    for (unsigned int i=0; i < nx; i++) {
        auto x = 0.5*lenx;
        auto y = leny - ( 0.5 + i )*cellSpacing;
        solution[i] = x*x-y*y;
    }
    for(int i = 0; i < solution.size(); i++)
    {
        auto j = nx/2 + nx*i;
        EXPECT_NEAR(u[j], solution[i],1e-5);
    }
}


// 2D Poisson Equation Test Case (Fitzpatrick Example)
//
// Problem Setup:
// - Domain: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1 (rectangular)
// - PDE: ∇²φ = f(x,y)
// - Source term: f(x,y) = 6xy (1-y) - 2x^3
//
// Boundary Conditions:
// - Dirichlet: φ(x,0) = φ(x,1) = 0, φ(0,y) = 0,  φ(1,y) = y(1-y)
//
// Analytical Solution:
// φ(x,y) = y*(1-y)x^3
//
// Reference: R. Fitzpatrick, "An example solution of Poisson's equation in 2-d"
// https://farside.ph.utexas.edu/teaching/329/lectures/node71.html
TEST_F(FVM_laplaceTests, 2DPoissonDerichlet) {

   //auto objReg = setUp(1, 161);
    auto objReg = setUp(1, 57);

    auto A = objReg.getSparseMatrixRef(AHandle);
    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);
    KERNEL::smatrix A1(A.rows(),A.columns());


    KERNEL::vector ae(nbCells, 0.0),aw(nbCells, 0.0),an(nbCells, 0.0),as(nbCells, 0.0),ap(nbCells, 0.0),sp(nbCells, 0.0),su(nbCells, 0.0);

    for (unsigned int i=0; i < nbCells; i++)
    {
        ae[i] = aw[i] = an[i] = as[i] = faceArea / cellSpacing;
        ap[i] = -4 * faceArea / cellSpacing;
        sp[i] = 0;
    }

// add souceTerm on all b value su, in the midle of the cell
    for (unsigned int jy = 0; jy < ny; ++jy) {
        double yP = leny - (0.5 + jy) * cellSpacing;

        for (unsigned int ix = 0; ix < nx; ++ix) {
            double xP = (0.5 + ix) * cellSpacing;
            unsigned int P = ix + jy * nx;

            double fP = 6.0 * xP * yP * (1.0 - yP) - 2.0 * xP * xP * xP;
            //double V  = faceArea;  // = cellSpacing*cellSpacing
            double V = faceArea * cellSpacing;

            b[P] += fP * V;
        }
    }

    // EAST
    for (unsigned int i = 0; i < ny; i++) {
        auto xpos = lenx;
        auto ypos = leny - (0.5+i)*cellSpacing;
        auto j = nx-1 +i*nx;
        ae[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * (ypos*(1-ypos)*xpos*xpos*xpos);
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    // NORTH
    for (unsigned int i = 0; i < nx; i++) {
        auto xpos = (0.5*cellSpacing + i*cellSpacing);
        auto ypos = leny;
        an[i] = 0.0;
        b[i] -= 2*faceArea/cellSpacing * 0;
        sp[i] -= 2*faceArea/cellSpacing;
        ap[i] -= faceArea/cellSpacing;
    }

    // WEST
    for (unsigned int i = 0; i < ny; i++) {
        auto xpos = 0.0;
        auto ypos = (leny - 0.5*cellSpacing - i * cellSpacing);
        auto j = i*nx;
        aw[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * 0;//(xpos*xpos-ypos*ypos);
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    // SOUTH
    for (unsigned int i = 0; i < nx; i++) {
        auto xpos = (0.5*cellSpacing + i*cellSpacing);
        auto ypos = 0.0;
        auto j = (ny-1)*nx + i;
        as[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * 0;//(xpos*xpos-ypos*ypos);
        sp[j] -= 2*faceArea/cellSpacing;
        ap[j] -= faceArea/cellSpacing;
    }

    buildMatrixWithBandsSpeed( A, ae, aw, as, an,sp, ap, nx);
    KERNEL::solve(A, u, b, 1e-15, 2000, KERNEL::BiCGSTAB);

    KERNEL::vector solution( nx, 0.0 );
    for (unsigned int i=0; i < nx; i++) {
        auto x = 0.5*lenx;
        auto y = leny - ( 0.5 + i )*cellSpacing;
        solution[i] = y*(1-y)*x*x*x;
    }

    double maxError = 0.0;
    int maxI = -1;   // index i i "solution"-vektoren (y-retning)
    int maxJ = -1;   // globalt index j i u-vektoren
    for(int i = 0; i < solution.size(); i++)
    {
        auto j = nx/2 + nx*i;

        EXPECT_NEAR(u[j], solution[i],1e-5);
    }
}