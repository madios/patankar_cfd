//
// Created by Peter Berg Ammundsen on 12/12/2025.
//

#ifndef UTIL_H
#define UTIL_H
#include <iomanip>
#include <iostream>
#include <string>
#include "GlobalTypeDefs.h"
#include "KernelTypeDefs.h"

namespace util {
    template< typename T >
    void printBlaze( const T& expr,
                     const std::string& name = "",
                     int width = 10,
                     int precision = 4 )
    {
        std::ostream& os = std::cout;
        const auto x = blaze::eval(expr);

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
        // Force evaluation to avoid expression-template lifetime issues and avoid deprecated operator~
        const auto a = blaze::eval( a_in );
        const auto b = blaze::eval( b_in );

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


KERNEL::dmatrix buildReferenceCoeffMatrix( GLOBAL::scalar A, GLOBAL::scalar dx, GLOBAL::scalar phi_b )
{
    const GLOBAL::scalar alpha = A / dx;

    KERNEL::dmatrix M( 9, 7, GLOBAL::scalar(0) );

    // Helper: column indices
    constexpr std::size_t c_aP = 0;
    constexpr std::size_t c_aW = 1;
    constexpr std::size_t c_aE = 2;
    constexpr std::size_t c_aS = 3;
    constexpr std::size_t c_aN = 4;
    constexpr std::size_t c_SU = 5;
    constexpr std::size_t c_SP = 6;

    // Cell 1: Internal
    M(0,c_aP) = -4.0 * alpha;
    M(0,c_aW) =  alpha;
    M(0,c_aE) =  alpha;
    M(0,c_aS) =  alpha;
    M(0,c_aN) =  alpha;
    M(0,c_SU) =  0.0;
    M(0,c_SP) =  0.0;

    // Cell 2: BC_W
    M(1,c_aP) = -5.0 * alpha;
    M(1,c_aW) =  0.0;
    M(1,c_aE) =  alpha;
    M(1,c_aS) =  alpha;
    M(1,c_aN) =  alpha;
    M(1,c_SU) = -2.0 * alpha * phi_b;
    M(1,c_SP) = -2.0 * alpha;

    // Cell 3: BC_E
    M(2,c_aP) = -5.0 * alpha;
    M(2,c_aW) =  alpha;
    M(2,c_aE) =  0.0;
    M(2,c_aS) =  alpha;
    M(2,c_aN) =  alpha;
    M(2,c_SU) = -2.0 * alpha * phi_b;
    M(2,c_SP) = -2.0 * alpha;

    // Cell 4: BC_S
    M(3,c_aP) = -5.0 * alpha;
    M(3,c_aW) =  alpha;
    M(3,c_aE) =  alpha;
    M(3,c_aS) =  0.0;
    M(3,c_aN) =  alpha;
    M(3,c_SU) = -2.0 * alpha * phi_b;
    M(3,c_SP) = -2.0 * alpha;

    // Cell 5: BC_N
    M(4,c_aP) = -5.0 * alpha;
    M(4,c_aW) =  alpha;
    M(4,c_aE) =  alpha;
    M(4,c_aS) =  alpha;
    M(4,c_aN) =  0.0;
    M(4,c_SU) = -2.0 * alpha * phi_b;
    M(4,c_SP) = -2.0 * alpha;

    // Cell 6: BC_WS
    M(5,c_aP) = -6.0 * alpha;
    M(5,c_aW) =  0.0;
    M(5,c_aE) =  alpha;
    M(5,c_aS) =  0.0;
    M(5,c_aN) =  alpha;
    M(5,c_SU) = -4.0 * alpha * phi_b;
    M(5,c_SP) = -4.0 * alpha;

    // Cell 7: BC_WN
    M(6,c_aP) = -6.0 * alpha;
    M(6,c_aW) =  0.0;
    M(6,c_aE) =  alpha;
    M(6,c_aS) =  alpha;
    M(6,c_aN) =  0.0;
    M(6,c_SU) = -4.0 * alpha * phi_b;
    M(6,c_SP) = -4.0 * alpha;

    // Cell 8: BC_ES
    M(7,c_aP) = -6.0 * alpha;
    M(7,c_aW) =  alpha;
    M(7,c_aE) =  0.0;
    M(7,c_aS) =  0.0;
    M(7,c_aN) =  alpha;
    M(7,c_SU) = -4.0 * alpha * phi_b;
    M(7,c_SP) = -4.0 * alpha;

    // Cell 9: BC_EN
    M(8,c_aP) = -6.0 * alpha;
    M(8,c_aW) =  alpha;
    M(8,c_aE) =  0.0;
    M(8,c_aS) =  alpha;
    M(8,c_aN) =  0.0;
    M(8,c_SU) = -4.0 * alpha * phi_b;
    M(8,c_SP) = -4.0 * alpha;

    return M;
}
};


#endif //UTIL_H
