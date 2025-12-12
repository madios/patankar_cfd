#pragma once
#include <blaze/Blaze.h>
#include "GlobalTypeDefs.h"

namespace KERNEL {
// #ifndef KERNEL_SCALAR_T
//     using scalar = double;              // default
// #else
//     using scalar = KERNEL_SCALAR_T;     // define this in cmake file
// #endif

    using vector = blaze::DynamicVector<GLOBAL::scalar, blaze::columnVector>;
    using smatrix = blaze::CompressedMatrix<GLOBAL::scalar, blaze::rowMajor>;
    using dmatrix = blaze::DynamicMatrix<GLOBAL::scalar, blaze::rowMajor>;
}
