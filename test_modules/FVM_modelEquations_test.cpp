#include <gtest/gtest.h>
#include "KERNEL.h"
#include <numeric>
#include "blaze/Blaze.h"
#include "structured2d.h"
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

    KERNEL::ObjectRegistry setUp(GLOBAL::scalar lengthX, unsigned int nbX,GLOBAL::scalar lengthY, unsigned int nbY)
    {
        if (nbX%2 == 0)
            throw std::runtime_error("ERROR: nx must be uneven");

        nx = nbX;
        ny = nbY;
        nbCells = nx*ny;
        leny = lengthX;
        lenx = lengthY;
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
    auto objReg = setUp(1.0, nbX,1.0,nbX);

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
// This test verifies that the solver correctly applies locally defined Dirichlet boundary conditions using a uniform mesh.
TEST_F(FVM_laplaceTests, FVM_localDerichletBCs) {

    auto objReg = setUp(3.0, 3,3.0,3);

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
enum class BoundaryFunction { single ,MultiplyXY };

void appliedDirichletBoundaryCondition(const MESH::RegionID region_id, KERNEL::vector& coef, KERNEL::vector& sp, KERNEL::vector& b, const MESH::structured2dRegularRectangle& mesh,const BoundaryFunction& boundaryFunction )
{
    GLOBAL::scalar boundaryValue = 0;
    const MESH::Region& region = mesh.region(region_id);
    for (auto cellId : region)
    {
        auto cellFacePos = mesh.getCellFacePos(region_id, cellId);
        switch (boundaryFunction)
        {
            case BoundaryFunction::MultiplyXY: boundaryValue =  cellFacePos.first*cellFacePos.second; break;
            case BoundaryFunction::single    : boundaryValue =  1; break;

        }
        coef[cellId]  = 0.0;
        b   [cellId] -= 2*mesh.getCellFaceAreal_X()/mesh.getCellSpacing_X() * boundaryValue;  // du har glemt at kigge på Y retningen afhængig af hvilken vej du kigger
        sp  [cellId] -= 2*mesh.getCellFaceAreal_X()/mesh.getCellSpacing_X();
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
// Same setup as FVM_localDirichletBCs, but here the mesh class is used and the Dirichlet boundary conditions are applied via appliedDirichletBoundaryCondition.
TEST_F(FVM_laplaceTests, FVM_localDerichletBCs_UsingMesh) {

    MESH::structured2dRegularRectangle mesh(1,  11,1,11);
    auto objReg = setUp(mesh.lenX(), mesh.nbCellsX(),mesh.lenY(), mesh.nbCellsY());
    auto A = objReg.getSparseMatrixRef(AHandle);
    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);

    KERNEL::vector ae(mesh.nbCells(), 0.0),aw(mesh.nbCells(), 0.0),an(mesh.nbCells(), 0.0),as(mesh.nbCells(), 0.0),ap(mesh.nbCells(), 0.0),sp(mesh.nbCells(), 0.0),su(mesh.nbCells(), 0.0);

    for (unsigned int i=0; i < mesh.nbCells(); i++)
    {
            ae[i] = aw[i] = an[i] = as[i] = mesh.getCellFaceAreal_X() / mesh.getCellSpacing_X();
    }
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_right ,ae,sp,b,mesh,BoundaryFunction::MultiplyXY);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_top   ,an,sp,b,mesh,BoundaryFunction::MultiplyXY);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_left  ,aw,sp,b,mesh,BoundaryFunction::MultiplyXY);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_bottom,as,sp,b,mesh,BoundaryFunction::MultiplyXY);
    buildMatrixWithBandsSpeed(A,ae,aw,as,an,sp,ap,nx);

    KERNEL::solve(A, u, b, 1e-10, 3000, KERNEL::BiCGSTAB);

    KERNEL::vector solution( nx, 0.0 );
    for (unsigned int i=0; i < nx; i++) {
        auto x = 0.5*mesh.lenX();
        auto y = mesh.lenY() - ( 0.5 + i )*mesh.getCellSpacing_Y();
        solution[i] = x*y;
    }

    for(int i = 0; i < solution.size(); i++)
    {
        auto j = nx/2 + nx*i;
        EXPECT_NEAR(u[j], solution[i],tolerance);
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
// Same setup as FVM_localDirichletBCs_UsingMesh, but here the mesh is a RectangularMesh instance.
// All boundaries is Dirichlet boundaries.
// ToDo There is  something wrong in this test, as the appliedDirichletBoundaryCondition only used dx and not dy
TEST_F(FVM_laplaceTests, FVM_localDerichletBCs_UsingMesh_RectangularMesh) {

    MESH::structured2dRegularRectangle mesh(1,51,2,101);
    auto objReg = setUp(mesh.lenX(), mesh.nbCellsX(),mesh.lenY(), mesh.nbCellsY());
    auto A = objReg.getSparseMatrixRef(AHandle);
    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);

    KERNEL::vector ae(mesh.nbCells(), 0.0),aw(mesh.nbCells(), 0.0),an(mesh.nbCells(), 0.0),as(mesh.nbCells(), 0.0),ap(mesh.nbCells(), 0.0),sp(mesh.nbCells(), 0.0),su(mesh.nbCells(), 0.0);
    ae = aw = an = as = mesh.getCellFaceAreal_X() / mesh.getCellSpacing_X();
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_right ,ae,sp,b,mesh,BoundaryFunction::MultiplyXY);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_top   ,an,sp,b,mesh,BoundaryFunction::MultiplyXY);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_left  ,aw,sp,b,mesh,BoundaryFunction::MultiplyXY);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_bottom,as,sp,b,mesh,BoundaryFunction::MultiplyXY);
    buildMatrixWithBandsSpeed(A,ae,aw,as,an,sp,ap,nx);

    KERNEL::solve(A, u, b, 1e-10, 2000, KERNEL::BiCGSTAB);

    KERNEL::vector solution( nx, 0.0 );
    for (unsigned int i=0; i < nx; i++) {
        auto x = 0.5*mesh.lenX();
        auto y = mesh.lenY() - ( 0.5 + i )*mesh.getCellSpacing_Y();
        solution[i] = x*y;
    }

    for(int i = 0; i < solution.size(); i++)
    {
        auto j = nx/2 + nx*i;
        EXPECT_NEAR(u[j], solution[i],tolerance);
    }
}

// Same as the test spacVarDerichletBCs, but here we use the function buildMatrixWithBandsSpeed to assemble the A-matrix in a more computationally efficient way.
// All boundaries is Dirichlet boundaries.
TEST_F(FVM_laplaceTests, spacVarDerichletBCsSpeed)
{

    //auto objReg = setUp(1, 161);
    auto objReg = setUp(1, 161,1, 161);

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
// This test verifies the solver’s ability to handle a more advanced Poisson system that includes a non-zero source term and an uniform mesh grid
// All boundaries is Dirichlet boundaries.
TEST_F(FVM_laplaceTests, 2DPoissonDerichlet) {

   //auto objReg = setUp(1, 161);
    auto objReg = setUp(1, 57,1,57);

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

    for(int i = 0; i < solution.size(); i++)
    {
        auto j = nx/2 + nx*i;

        EXPECT_NEAR(u[j], solution[i],1e-5);
    }
}



//2D Laplace med blandede BCs
//
//Domæne:
//  0 ≤ x ≤ 1, 0 ≤ y ≤ 1

//PDE:
//  ∇²φ = 0

//BC:
//  y = 0:  φ(x,0)   = 0          (Dirichlet)
//  y = 1:  φ(x,1)   = 1          (Dirichlet)
//  x = 0:  ∂φ/∂x|x=0 = 0         (Neumann, nul gradient)
//  x = 1:  ∂φ/∂x|x=1 = 0         (Neumann, nul gradient)
//
//Analytisk løsning:
//  φ(x,y) = y
// The purpose of this test is to verify that the solver correctly handles simple Neumann boundary conditions,
// where zero normal gradients at x=0 and x=1.
TEST_F(FVM_laplaceTests, 2DLaplaceWithDirichletAndNuemannBC)
{


}

//2D Poisson equation with mixed Dirichlet/Neumann BCs
//Domain: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1
//PDE:
//  ∇²φ = 2x
//
//Analytical solution:
//  φ(x,y) = x * y²
//  φ(x,y) =( -x² + 2 * x ) * y
//
//Boundary conditions:
//  y = 0:  φ(x,0) = 0*x                 (Dirichlet)
//  x = 0:  φ(0,y) = 0*y                 (Dirichlet)
//  x = 1:  φ(x,1) = x * ( 1 - x ) + X   (Dirichlet)
//  y = 1:  ∂φ/∂x|x=1 = 0                (Neumann)
// The purpose of this test is to verify that the solver correctly handles more advanced Neumann boundary conditions,
// where zero normal gradients x=1.
TEST_F(FVM_laplaceTests, 2DLaplaceWithMixedDirichletAndNuemannBC) {
}

TEST_F(FVM_laplaceTests, DiffusionCoefficientMatrixTest) {
    MESH::structured2dRegularRectangle mesh(3, 3, 3, 3);
    auto objReg = setUp(mesh.lenX(), mesh.nbCellsX(), mesh.lenY(), mesh.nbCellsY());
    auto A = objReg.getSparseMatrixRef(AHandle);
    auto u = objReg.getVectorRef(uHandle);
    auto b = objReg.getVectorRef(bHandle);

    KERNEL::vector ae(mesh.nbCells(), 0.0),aw(mesh.nbCells(), 0.0),an(mesh.nbCells(), 0.0),as(mesh.nbCells(), 0.0),ap(mesh.nbCells(), 0.0),sp(mesh.nbCells(), 0.0),su(mesh.nbCells(), 0.0);
    ae = aw = an = as = mesh.getCellFaceAreal_X() / mesh.getCellSpacing_X();
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_right ,ae,sp,b,mesh,BoundaryFunction::single);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_top   ,an,sp,b,mesh,BoundaryFunction::single);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_left  ,aw,sp,b,mesh,BoundaryFunction::single);
    appliedDirichletBoundaryCondition(MESH::RegionID::Boundary_bottom,as,sp,b,mesh,BoundaryFunction::single);
    buildMatrixWithBandsSpeed(A,ae,aw,as,an,sp,ap,nx);
    GLOBAL::scalar phi_b = 2;

    KERNEL::dmatrix refMatrix = util::buildReferenceCoeffMatrix( mesh.getCellFaceAreal_X(), mesh.getCellSpacing_X(), phi_b );


//    std::cout<<"refMatrix = \n"<<refMatrix<<std::endl;
//    std::cout<<"A = \n"<<A<<std::endl;
    //Test AP
    EXPECT_EQ( refMatrix(0,0), A(4,4) );   // internal
    EXPECT_EQ( refMatrix(1,0), A(3,3) );   // BC_w
    EXPECT_EQ( refMatrix(2,0), A(5,5) );   // BC_E
    EXPECT_EQ( refMatrix(3,0), A(7,7) );   // BC_s
    EXPECT_EQ( refMatrix(4,0), A(1,1) );   // BC_N
    EXPECT_EQ( refMatrix(5,0), A(6,6) );   // BC_SW
    EXPECT_EQ( refMatrix(6,0), A(0,0) );   // BC_NW
    EXPECT_EQ( refMatrix(7,0), A(8,8) );   // BC_SE
    EXPECT_EQ( refMatrix(8,0), A(2,2) );   // BC_NE

    //Test AW
    EXPECT_EQ( refMatrix(0,1), A(3,4) );   // internal
    EXPECT_EQ( refMatrix(1,1), A(2,3) );   // BC_w
    EXPECT_EQ( refMatrix(2,1), A(4,5) );   // BC_E
    EXPECT_EQ( refMatrix(3,1), A(6,7) );   // BC_s
    EXPECT_EQ( refMatrix(4,1), A(0,1) );   // BC_N
    EXPECT_EQ( refMatrix(5,1), A(5,6) );   // BC_SW
    //EXPECT_EQ( refMatrix(6,1), A(0,0) );   // BC_NW
    EXPECT_EQ( refMatrix(7,1), A(7,8) );   // BC_SE
    EXPECT_EQ( refMatrix(8,1), A(1,2) );   // BC_NE

    //Test AE
    EXPECT_EQ( refMatrix(0,2), A(4+1,4) );   // internal
    EXPECT_EQ( refMatrix(1,2), A(3+1,3) );   // BC_w
    EXPECT_EQ( refMatrix(2,2), A(5+1,5) );   // BC_E
    EXPECT_EQ( refMatrix(3,2), A(7+1,7) );   // BC_s
    EXPECT_EQ( refMatrix(4,2), A(1+1,1) );   // BC_N
    EXPECT_EQ( refMatrix(5,2), A(6+1,6) );   // BC_SW
    EXPECT_EQ( refMatrix(6,2), A(0+1,0) );   // BC_NW
    EXPECT_EQ( refMatrix(7,2), A(8+1,8) );   // BC_SE
    EXPECT_EQ( refMatrix(8,2), A(2+1,2) );   // BC_NE

    //Test AS
    EXPECT_EQ( refMatrix(0,3), A(4+3,4) );   // internal
    EXPECT_EQ( refMatrix(1,3), A(3+3,3) );   // BC_w
    EXPECT_EQ( refMatrix(2,3), A(5+3,5) );   // BC_E
    //EXPECT_EQ( refMatrix(3,3), A(7+3,7) );   // BC_s
    EXPECT_EQ( refMatrix(4,3), A(1+3,1) );   // BC_N
    EXPECT_EQ( refMatrix(5,3), A(6+3,6) );   // BC_SW
    EXPECT_EQ( refMatrix(6,3), A(0+3,0) );   // BC_NW
    //EXPECT_EQ( refMatrix(7,3), A(8+3,8) );   // BC_SE
    EXPECT_EQ( refMatrix(8,3), A(2+3,2) );   // BC_NE
}