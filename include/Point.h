#ifndef MATH_LIBRARY_POINT
#define MATH_LIBRARY_POINT

#include "Vector.h"

namespace math {

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
using Point = Vector<T, Size, Option>;

// Point of floats
template <unsigned int Size, Options Option = Options::RowMajor>
using PointNf = Point<float, Size, Option>;

// Point of doubles
template <unsigned int Size, Options Option = Options::RowMajor>
using PointNd = Point<double, Size, Option>;

// Point of ints
template <unsigned int Size, Options Option = Options::RowMajor>
using PointNi = Point<int, Size, Option>;

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

using Point2Di = Point2D<int>;
using Point3Di = Point3D<int>;
using Point4Di = Point4D<int>;

}  // namespace math

#endif