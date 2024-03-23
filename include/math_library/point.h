/**
 * @file point.h
 */

#ifndef MATH_LIBRARY_POINT_H
#define MATH_LIBRARY_POINT_H

#include "vector.h"

namespace math {

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
using Point = Vector<T, Size, Option>;

// Point of floats
template <std::size_t Size, Options Option = Options::RowMajor>
using PointNf = Point<float, Size, Option>;

// Point of doubles
template <std::size_t Size, Options Option = Options::RowMajor>
using PointNd = Point<double, Size, Option>;

// Point of ints
template <std::size_t Size, Options Option = Options::RowMajor>
using PointNi = Point<std::int32_t, Size, Option>;

// Templated Point 2D
template <typename T, Options Option = Options::RowMajor>
using Point2D = Point<T, 2, Option>;

// Templated Point 3D
template <typename T, Options Option = Options::RowMajor>
using Point3D = Point<T, 3, Option>;

// Templated Point 4D
template <typename T, Options Option = Options::RowMajor>
using Point4D = Point<T, 4, Option>;

// Specific data type points
using Point2Df = Point2D<float>;
using Point3Df = Point3D<float>;
using Point4Df = Point4D<float>;

using Point2Dd = Point2D<double>;
using Point3Dd = Point3D<double>;
using Point4Dd = Point4D<double>;

using Point2Di = Point2D<std::int32_t>;
using Point3Di = Point3D<std::int32_t>;
using Point4Di = Point4D<std::int32_t>;

}  // namespace math

#endif