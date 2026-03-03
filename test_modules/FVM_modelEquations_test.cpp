#include <gtest/gtest.h>
#include "KERNEL.h"
#include <numeric>
#include "Util.h"

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

    KERNEL::ObjectRegistry setUp(GLOBAL::scalar length, unsigned int nbX)
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

    auto correctResult = *KERNEL::vector(b.size());
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
// - Domain: 0 ≤ x ≤ lx, 0 ≤ y ≤ ly (rectangular)
// - ODE: ∇²φ = 0
//
// Boundary Conditions:
// - Dirichlet: φ(x,0) = 0, φ(0,y) = 0, φ(x,ly) = x*ly,  φ(lx,y) = y*lx
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
    }

    // EAST
    for (unsigned int i = 0; i < ny; i++) {
        auto xpos = lenx;
        auto ypos = leny - (0.5+i)*cellSpacing;
        auto j = nx-1 +i*nx;
        ae[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * ypos*xpos;
        ap[j] -= faceArea/cellSpacing;
    }

    // NORTH
    for (unsigned int i = 0; i < nx; i++) {
        auto xpos = (0.5*cellSpacing + i*cellSpacing);
        auto ypos = leny;
        an[i] = 0.0;
        b[i] -= 2*faceArea/cellSpacing * xpos*ypos;
        ap[i] -= faceArea/cellSpacing;
    }

    // WEST
    for (unsigned int i = 0; i < ny; i++) {
        auto xpos = 0.0;
        auto ypos = (leny - 0.5*cellSpacing - i * cellSpacing);
        auto j = i*nx;
        aw[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * xpos*ypos;
        ap[j] -= faceArea/cellSpacing;
    }

    // SOUTH
    for (unsigned int i = 0; i < nx; i++) {
        auto xpos = (0.5*cellSpacing + i*cellSpacing);
        auto ypos = 0.0;
        auto j = (ny-1)*nx + i;
        as[j] = 0.0;
        b[j] -= 2*faceArea/cellSpacing * xpos*ypos;
        ap[j] -= faceArea/cellSpacing;
    }

    for (unsigned int i = 0; i < ap.size(); i++) {
        blaze::band(A,0)[i] = ap[i];
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

    KERNEL::solve(A, u, b, 1e-15, 1000, KERNEL::BiCGSTAB);

    // theoretical solution, vertical mid-line at x = lenx/2
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

// This test work directly on Blaze band views.
// This implementation is quite fast: ~3 s runtime for a 301×301 system in a Release build.
// Reserving memory for the sparse matrix takes ~2.5 seconds, but once reserved the matrix assembly itself
// takes only ~2 ms (when reusing the same A matrix is possible).
// with out reserving is the execution time ~6 seconds
TEST_F(FVM_laplaceTests, FVM_localDerichletBCs3)
{
    unsigned int nbX = 11.0;
    auto objReg = setUp(1.0, nbX);

    auto A = objReg.getSparseMatrixRef(AHandle);
    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);

    auto ap = blaze::band(A,0);
    auto ae = blaze::band(A,1);
    auto aw = blaze::band(A,-1);
    auto as = blaze::band(A,nx);
    auto an = blaze::band(A,static_cast<int>(-nx));

    // reserve in sparse matrix.
    A.reserve(nbCells * 5);
    for (std::size_t r = 0; r < nbCells; ++r) A.reserve(r, 5);

    // Collect A
    const GLOBAL::scalar pVal = -4.0 * faceArea / cellSpacing;
    const GLOBAL::scalar fVal =  1.0 * faceArea / cellSpacing;

    //Main Diagonal
    for (size_t i = 0; i < ap.size(); ++i) {ap[i] = pVal;}
    for (auto p : cellIndices_West)  {ap[p] -= fVal;}
    for (auto p : cellIndices_East)  {ap[p] -= fVal;}
    for (auto p : cellIndices_North) {ap[p] -= fVal;}
    for (auto p : cellIndices_South) {ap[p] -= fVal;}

    //East diagonal
    for (size_t i = 0; i < ap.size()-1; ++i){ae[i]   = fVal;}
    for (size_t i = 0; i < cellIndices_East.size()-1; ++i){ae[cellIndices_East[i]]   = 0.0;}

    //West diagonal
    for (size_t i = 0; i < ap.size()-1; ++i){aw[i]   = fVal;}
    for (size_t i = 1; i < cellIndices_West.size(); ++i) {aw[cellIndices_West[i]-1]   = 0.0;}

    //South
    for (size_t i = 0; i < ap.size()-nx; ++i){as[i]   = fVal;}

    //North
    for (size_t i = 0; i < ap.size()-nx; ++i){an[i]   = fVal;}
    // Collect b for Dirichlet boundaries
    const GLOBAL::scalar bVal = 2.0 * faceArea / cellSpacing;

    // NORTH
    GLOBAL::scalar ypos = leny;
    for( std::size_t i = 0; i < nx; ++i ) {
        const std::size_t p = i;
        const GLOBAL::scalar xpos = (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * xpos * ypos;
    }

    // SOUTH
    ypos = 0.0;
    for( std::size_t i = 0; i < nx; ++i ) {
        const std::size_t p = (ny-1) * nx + i;
        const GLOBAL::scalar xpos = (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * xpos * ypos;
    }

    // EAST
    GLOBAL::scalar xpos = lenx;
    for( std::size_t i = 0; i < ny; ++i ) {
        const std::size_t p = i * nx + (nx-1);
        const GLOBAL::scalar ypos = leny - (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * xpos * ypos;
    }

    // WEST
    xpos = 0.0;
    for( std::size_t i = 0; i < ny; ++i ) {
        const std::size_t p = i * nx;
        const GLOBAL::scalar ypos = leny - (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * xpos * ypos;
    }
    KERNEL::solve(A, u, b, 1e-15, 100000, KERNEL::BiCGSTAB);

    // Theoretical solution, vertical mid-line at x = lenx/2
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


// 2D Laplace Equation with spacially varying BCs
//
// Problem Setup:
// - Domain: 0 ≤ x ≤ lx, 0 ≤ y ≤ ly (rectangular)
// - ODE: ∇²φ = 0
//
// Boundary Conditions:
// - Dirichlet: φ(x,0) = xˆ2 , φ(0,y) = -yˆ2, φ(x,ly) = xˆ2-ly^2,  φ(lx,y) = lx^2-yˆ2
//
// Analytical Solution:
// φ(x,y) = xˆ2-yˆ2
TEST_F(FVM_laplaceTests, sparseVarDerichletBCsSpeed)
{

    //auto objReg = setUp(1, 161);
    auto objReg = setUp(1, 161);

    auto A = objReg.getSparseMatrixRef(AHandle);
    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);

    auto ap = blaze::band(A,0);
    auto ae = blaze::band(A,1);
    auto aw = blaze::band(A,-1);
    auto as = blaze::band(A,nx);
    auto an = blaze::band(A,static_cast<int>(-nx));

    // reserve in sparse matrix.
    A.reserve(nbCells * 5);
    for (std::size_t r = 0; r < nbCells; ++r) A.reserve(r, 5);

    // Collect A
    const GLOBAL::scalar pVal = -4.0 * faceArea / cellSpacing;
    const GLOBAL::scalar fVal =  1.0 * faceArea / cellSpacing;

    //Main Diagonal
    for (size_t i = 0; i < ap.size(); ++i) {ap[i] = pVal;}
    for (auto p : cellIndices_West)  {ap[p] -= fVal;}
    for (auto p : cellIndices_East)  {ap[p] -= fVal;}
    for (auto p : cellIndices_North) {ap[p] -= fVal;}
    for (auto p : cellIndices_South) {ap[p] -= fVal;}

    //East diagonal
    for (size_t i = 0; i < ap.size()-1; ++i){ae[i]   = fVal;}
    for (size_t i = 0; i < cellIndices_East.size()-1; ++i){ae[cellIndices_East[i]]   = 0.0;}

    //West diagonal
    for (size_t i = 0; i < ap.size()-1; ++i){aw[i]   = fVal;}
    for (size_t i = 1; i < cellIndices_West.size(); ++i) {aw[cellIndices_West[i]-1]   = 0.0;}

    //South
    for (size_t i = 0; i < ap.size()-nx; ++i){as[i]   = fVal;}

    //North
    for (size_t i = 0; i < ap.size()-nx; ++i){an[i]   = fVal;}
    // Collect b for Dirichlet boundaries
    const GLOBAL::scalar bVal = 2.0 * faceArea / cellSpacing;

    // NORTH
    GLOBAL::scalar ypos = leny;
    for( std::size_t i = 0; i < nx; ++i ) {
        const std::size_t p = i;
        const GLOBAL::scalar xpos = (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * (xpos * xpos - ypos * ypos);
    }

    // SOUTH
    ypos = 0.0;
    for( std::size_t i = 0; i < nx; ++i ) {
        const std::size_t p = (ny-1) * nx + i;
        const GLOBAL::scalar xpos = (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * (xpos * xpos - ypos * ypos);
    }

    // EAST
    GLOBAL::scalar xpos = lenx;
    for( std::size_t i = 0; i < ny; ++i ) {
        const std::size_t p = i * nx + (nx-1);
        const GLOBAL::scalar ypos = leny - (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * (xpos * xpos - ypos * ypos);
    }

    // WEST
    xpos = 0.0;
    for( std::size_t i = 0; i < ny; ++i ) {
        const std::size_t p = i * nx;
        const GLOBAL::scalar ypos = leny - (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * (xpos * xpos - ypos * ypos);
    }

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
// - Domain: 0 ≤ x ≤ lx, 0 ≤ y ≤ lx (rectangular)
// - PDE: ∇²φ = f(x,y)
// - Source term: f(x,y) = 6xy (1-y) - 2x^3
//
// Boundary Conditions:
// - Dirichlet: φ(x,0) = φ(x,1) = 0, φ(0,y) = 0,  φ(1,y) = y(1-y) * lx^3
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

    auto ap = blaze::band(A,0);
    auto ae = blaze::band(A,1);
    auto aw = blaze::band(A,-1);
    auto as = blaze::band(A,nx);
    auto an = blaze::band(A,static_cast<int>(-nx));

    // reserve in sparse matrix.
    A.reserve(nbCells * 5);
    for (std::size_t r = 0; r < nbCells; ++r) A.reserve(r, 5);

    // Collect A
    const GLOBAL::scalar pVal = -4.0 * faceArea / cellSpacing;
    const GLOBAL::scalar fVal =  1.0 * faceArea / cellSpacing;

    //Main Diagonal
    for (size_t i = 0; i < ap.size(); ++i) {ap[i] = pVal;}
    for (auto p : cellIndices_West)  {ap[p] -= fVal;}
    for (auto p : cellIndices_East)  {ap[p] -= fVal;}
    for (auto p : cellIndices_North) {ap[p] -= fVal;}
    for (auto p : cellIndices_South) {ap[p] -= fVal;}

    //East diagonal
    for (size_t i = 0; i < ap.size()-1; ++i){ae[i]   = fVal;}
    for (size_t i = 0; i < cellIndices_East.size()-1; ++i){ae[cellIndices_East[i]]   = 0.0;}

    //West diagonal
    for (size_t i = 0; i < ap.size()-1; ++i){aw[i]   = fVal;}
    for (size_t i = 1; i < cellIndices_West.size(); ++i) {aw[cellIndices_West[i]-1]   = 0.0;}

    //South
    for (size_t i = 0; i < ap.size()-nx; ++i){as[i]   = fVal;}

    //North
    for (size_t i = 0; i < ap.size()-nx; ++i){an[i]   = fVal;}
    // Collect b for Dirichlet boundaries
    const GLOBAL::scalar bVal = 2.0 * faceArea / cellSpacing;

    // Now we have a source term, so we need to assign source values to all entries in b
    for (unsigned int jy = 0; jy < ny; ++jy) {
        double yP = leny - (0.5 + jy) * cellSpacing;

        for (unsigned int ix = 0; ix < nx; ++ix) {
            double xP = (0.5 + ix) * cellSpacing;
            unsigned int P = ix + jy * nx;

            double fP = 6.0 * xP * yP * (1.0 - yP) - 2.0 * xP * xP * xP;
            double V = faceArea * cellSpacing;

            b[P] += fP * V;
        }
    }

    // NORTH
    GLOBAL::scalar ypos = leny;
    for( std::size_t i = 0; i < nx; ++i ) {
        const std::size_t p = i;
        const GLOBAL::scalar xpos = (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * (ypos*(1-ypos)*xpos*xpos*xpos);
    }

    // SOUTH
    ypos = 0.0;
    for( std::size_t i = 0; i < nx; ++i ) {
        const std::size_t p = (ny-1) * nx + i;
        const GLOBAL::scalar xpos = (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * (ypos*(1-ypos)*xpos*xpos*xpos);
    }

    // EAST
    GLOBAL::scalar xpos = lenx;
    for( std::size_t i = 0; i < ny; ++i ) {
        const std::size_t p = i * nx + (nx-1);
        const GLOBAL::scalar ypos = leny - (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * (ypos*(1-ypos)*xpos*xpos*xpos);
    }

    // WEST
    xpos = 0.0;
    for( std::size_t i = 0; i < ny; ++i ) {
        const std::size_t p = i * nx;
        const GLOBAL::scalar ypos = leny - (0.5 + static_cast<GLOBAL::scalar>(i)) * cellSpacing;
        b[p] -= bVal * (ypos*(1-ypos)*xpos*xpos*xpos);
    }

    //buildMatrixWithBandsSpeed( A, ae, aw, as, an,sp, ap, nx);
    KERNEL::solve(A, u, b, 1e-15, 2000, KERNEL::BiCGSTAB);

    KERNEL::vector solution( nx, 0.0 );
    for (unsigned int i=0; i < nx; i++) {
        auto x = 0.5*lenx;
        auto y = leny - ( 0.5 + i )*cellSpacing;
        solution[i] = y*(1-y)*x*x*x;
    }

    for(int i = 0; i < solution.size(); i++)
    {
        auto j = nx/2 + nx*i;

        EXPECT_NEAR(u[j], solution[i],1e-5);
    }
}