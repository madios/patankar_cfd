//#include "../include/KERNEL.h"
#include "KERNEL.h"
#include "GlobalTypeDefs.h"
#include "LinEqsSolvers.h"
#include "blaze/Blaze.h"

// make object registry to have own files.
KERNEL::ObjectRegistry::ObjectRegistry() = default;
KERNEL::ObjectRegistry::~ObjectRegistry() = default;
KERNEL::ObjectRegistry::ObjectRegistry(ObjectRegistry&&) noexcept = default;
KERNEL::ObjectRegistry& KERNEL::ObjectRegistry::operator=(ObjectRegistry&&) noexcept = default;

// Vector creation
KERNEL::VectorHandle KERNEL::ObjectRegistry::newVector(size_t size, GLOBAL::scalar initialValue) {
    if (registryClosed_)
        throw std::runtime_error("Registry closed. New objects must be defined before closing registry.");

    auto id = nextID_++;

    registry_[id] = std::make_unique<KERNEL::vector>(size, initialValue);

    return VectorHandle{id};
}

KERNEL::MatrixHandle KERNEL::ObjectRegistry::newMatrix(size_t rows, size_t cols, bool sparse) {
    if (registryClosed_)
        throw std::runtime_error("Registry closed. New objects must be defined before closing registry.");

    auto id = nextID_++;
    // nextID_ = nextID_ + 1;
    // auto id = nextID_;

    if (sparse)
        registry_[id] = std::make_unique<KERNEL::smatrix>(rows, cols);
    else
        registry_[id] = std::make_unique<KERNEL::dmatrix>(rows, cols);

    return MatrixHandle{id};
}

KERNEL::vector& KERNEL::ObjectRegistry::getVectorRef(VectorHandle handle) {
    if (!registryClosed_) {
        throw std::runtime_error("Close registry before accessing objects.");
    }
    auto it = registry_.find(handle.id);
    if (it == registry_.end()) {
        throw std::runtime_error("Invalid object ID");
    }

    // Try to get vector from variant
    std::unique_ptr<KERNEL::vector>* vecPtr = std::get_if< std::unique_ptr<KERNEL::vector>>(&it->second);
    if (!vecPtr) {
        throw std::runtime_error("Object is not a vector");
    }

    return **vecPtr;
}

KERNEL::dmatrix& KERNEL::ObjectRegistry::getDenseMatrixRef(MatrixHandle handle) {

    auto it = registry_.find(handle.id);

    if (it == registry_.end())
        throw std::runtime_error("Invalid object ID");

    auto* matPtr = std::get_if<std::unique_ptr<KERNEL::dmatrix>>(&it->second);
    if (!matPtr)
        throw std::runtime_error("Object is not a dense matrix");

    return **matPtr;
}

KERNEL::smatrix& KERNEL::ObjectRegistry::getSparseMatrixRef(MatrixHandle handle) {

    auto it = registry_.find(handle.id);

    if (it == registry_.end())
        throw std::runtime_error("Invalid object ID");

    auto* matPtr = std::get_if<std::unique_ptr<KERNEL::smatrix>>(&it->second);
    if (!matPtr)
        throw std::runtime_error("Object is not a sparse matrix");

    return **matPtr;
}

std::shared_ptr<KERNEL::vector> KERNEL::newTempVector(size_t size) {
    return std::make_shared<KERNEL::vector>(size, 0.0);
}


void KERNEL::solve(const KERNEL::dmatrix& A, KERNEL::vector& x, const KERNEL::vector& b, const GLOBAL::scalar tolerance, const unsigned int maxIter, KERNEL::SolverMethod method) {

    // static_assert( std::is_same_v<decltype(A), const KERNEL::dmatrix& >, "Error in KERNEL::solve: input matrix not dense.");

    checkLinEqSystemConsistency(A,b);

    if (method == BiCGSTAB) {
        LINEQSOLVERS::solve_BiCGSTAB(A, x, b, tolerance, maxIter);
    }else if (method == GaussSeidel) {
        LINEQSOLVERS::solve_GaussSeidel(A, x, b, tolerance, maxIter);
    }else if (method == Jacobi) {
        LINEQSOLVERS::solve_Jacobi(A, x, b, tolerance, maxIter);
    }else if (method == Blaze_automatic) {
        blaze::solve(A, x, b);
    }
}

void KERNEL::solve(const KERNEL::smatrix& A, KERNEL::vector& x, const KERNEL::vector& b, const GLOBAL::scalar tolerance, const unsigned int maxIter, KERNEL::SolverMethod method) {

    // static_assert( std::is_same_v<decltype(A), const KERNEL::smatrix& >, "Error in KERNEL::solve: input matrix not sparse.");

    checkLinEqSystemConsistency(A,b);

    if (method == BiCGSTAB) {
        LINEQSOLVERS::solve_BiCGSTAB(A, x, b, tolerance, maxIter);
    }

}
