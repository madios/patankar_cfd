//
// Created by Peter Berg Ammundsen on 08/12/2025.
//

#ifndef STRUCTURED2D_H
#define STRUCTURED2D_H
#pragma once

#include <stdexcept>
#include <unordered_map>
#include "GlobalTypeDefs.h"



namespace MESH {

    // Enum for region identifiers
    enum class RegionID {
        Entire,
        Boundary_left,
        Boundary_bottom,
        Boundary_right,
        Boundary_top
    };

    struct Region {
        struct iterator {
            using value_type        = int;
            using difference_type   = std::ptrdiff_t;
            using reference         = int;            // yields by value
            using pointer           = const int*;     // not dereferenced normally
            using iterator_category = std::forward_iterator_tag;

            int current;
            int step;
            int remaining;

            reference operator*() const { return current; }
            iterator& operator++() { current += step; --remaining; return *this; }
            iterator operator++(int) { auto tmp = *this; ++(*this); return tmp; }
            bool operator==(const iterator& other) const { return remaining == other.remaining; }
            bool operator!=(const iterator& other) const { return remaining != other.remaining; }
        };

        int start;
        int step;
        int count;

        iterator begin() const { return { start, step, count }; }
        iterator end()   const { return { 0, step, 0 }; }
        iterator last()  const { return { start + step*(count - 1), step, 1 }; }
    };


    class structured2dRegularRectangle{

    private:
        // order must follow initialization list
        const float lenX_, lenY_;
        const unsigned int nbCellsX_, nbCellsY_, nbCells_;

        // Registry of all regions; enum class ensures type safety
        std::unordered_map<RegionID, Region> regions;

        // how can I do this at compile time?
        // Each boundary side is represented by a vector of cell indices
        // example: rectangular mesh with (nbX,nbY)=(5,4) cells
        // its cell and boundary indices are illustrated here:
        //
        //               3
        //     +----+----+----+----+----+
        //     |  0 |  1 |  2 |  3 |  4 |
        //     +----+----+----+----+----+
        //     |  5 |  6 |  7 |  8 |  9 |
        //  0  +----+----+----+----+----+    2
        //     | 10 | 11 | 12 | 13 | 14 |
        //     +----+----+----+----+----+
        //     | 15 | 16 | 17 | 18 | 19 |
        //     +----+----+----+----+----+
        //               1
        //
        // the boundaries are names counter-clockwise, starting with the left one.
        // the cell ids per boundary also follow a counter-clockwise direction.
        // boundaries[0] = [0,5,10,15]
        // boundaries[1] = [15,16,17,18,19]
        // boundaries[2] = [19,14,9,4]
        // boundaries[3] = [4,3,2,1,0]

        void fillRegion(RegionID region);

    public:
        structured2dRegularRectangle(float lengthX, unsigned int nbCellsX, float lengthY, unsigned int nbCellsY);

        unsigned int nbCellsX() const { return nbCellsX_; }
        unsigned int nbCellsY() const { return nbCellsY_; }
        unsigned int nbCells() const { return nbCells_; }
        GLOBAL::scalar lenX() const { return lenX_; }
        GLOBAL::scalar lenY() const { return lenY_; }

        const GLOBAL::scalar getCellSpacing_X( ) const
        {
            return lenX_/static_cast<GLOBAL::scalar>(nbCellsX_);
        }
        const GLOBAL::scalar getCellSpacing_Y( ) const
        {
            return lenY_/static_cast<GLOBAL::scalar>(nbCellsY_);
        }

        // returns a cell 1/dx and 1/dy.
        // in regular meshes, these are same everywhere
        // should accept iterator to mesh cell
        const GLOBAL::scalar getCellReciprocalSpacing_X( ) const {
            return static_cast<GLOBAL::scalar>(nbCellsX_)/lenX_;
        }

        const GLOBAL::scalar getCellReciprocalSpacing_Y( ) const {
            return static_cast<GLOBAL::scalar>(nbCellsY_)/lenY_;
        }

        const GLOBAL::scalar getCellFaceAreal_Y( ) const
        {
            return getCellReciprocalSpacing_Y()*getCellReciprocalSpacing_Y();
        }

        const GLOBAL::scalar getCellFaceAreal_X( ) const
        {
            return getCellReciprocalSpacing_X()*getCellReciprocalSpacing_X();
        }

        // OBS!! This is for node-centered CV. Will change to face-centered CV
        const GLOBAL::scalar getCellCenterCoordinate_X(int cellId) const;
        const GLOBAL::scalar getCellCenterCoordinate_Y(int cellId) const;

        const Region& region(RegionID id) const {
            auto it = regions.find(id);
            if (it == regions.end()) throw std::out_of_range("Unknown region id");
            return it->second;
        }
        bool isBoundaryCell( unsigned int i ) const;
        const std::pair<GLOBAL::scalar,GLOBAL::scalar> getCellFacePos(RegionID id, int cellID) const;
    };

}
#endif //STRUCTURED2D_H
