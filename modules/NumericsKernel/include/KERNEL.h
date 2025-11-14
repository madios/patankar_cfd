#include <unordered_map>
#include <variant>
#include <memory>
#include "KernelTypeDefs.h"

namespace KERNEL {
    enum SolverMethod {
        GaussSeidel, BiCGSTAB, Jacobi, Blaze_automatic
    };

    // Variant holding unique_ptrs to vector or matrix
    using ObjectVariantPtr = std::variant<
        std::unique_ptr<KERNEL::vector>,
        std::unique_ptr<KERNEL::dmatrix>,
        std::unique_ptr<KERNEL::smatrix>
    >;

    struct VectorHandle { size_t id; };
    struct MatrixHandle { size_t id; };

    std::shared_ptr<KERNEL::vector> newTempVector(size_t size);

    void solve(const KERNEL::smatrix& A, KERNEL::vector& x, const KERNEL::vector& b, const KERNEL::scalar tolerance, const unsigned int maxIter, SolverMethod method);
    void solve(const KERNEL::dmatrix& A, KERNEL::vector& x, const KERNEL::vector& b, const KERNEL::scalar tolerance, const unsigned int maxIter, SolverMethod method);

    class ObjectRegistry {
    private:
        std::unordered_map<size_t, ObjectVariantPtr> registry_;
        size_t nextID_ = 0;
        bool registryClosed_ = false;

    public:

        // I am closing the reg after initial problem setup.
        // I can therefore savely return references to its objects
        // from associated getter functions
        void closeRegistry() {
            registryClosed_ = true;
            registry_.reserve(registry_.size());  // Prevent reallocation
        }

        // definition must be together with blaze includes.
        // Here, together with blaz's forward declaration,
        // I can only declare:
        ObjectRegistry();
        ~ObjectRegistry();
        ObjectRegistry(ObjectRegistry&&) noexcept;
        ObjectRegistry& operator=(ObjectRegistry&&) noexcept;

        // deleting copy constructor:
        ObjectRegistry(const ObjectRegistry&) = delete;
        ObjectRegistry& operator=(const ObjectRegistry&) = delete;

        VectorHandle newVector(size_t size, double initialValue = 0.0);
        MatrixHandle newMatrix(size_t rows, size_t cols, bool sparse = false );


        // ============ Getters ===================

        // these object references are valid, because I cannot create or
        // delete new objects in the registry after it is closed.

        KERNEL::vector& getVectorRef(VectorHandle handle);

        KERNEL::dmatrix& getDenseMatrixRef(MatrixHandle handle);
        KERNEL::smatrix& getSparseMatrixRef(MatrixHandle handle);

    };
}