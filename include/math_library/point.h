/**
 * @file point.h
 * @brief Defines various templated Point types for 2D, 3D, and 4D points with
 * different data types.
 *
 * This file provides templates for creating points of different dimensions (2D,
 * 3D, 4D) and with different data types (float, double, int, unsigned int).
 * These types are used to represent points in a mathematical space.
 */

#ifndef MATH_LIBRARY_POINT_H
#define MATH_LIBRARY_POINT_H

#include "vector.h"

namespace math {

/**
 * @brief Defines a Point as a specialized Vector.
 *
 * This alias defines a Point as a specialized Vector with the same
 * dimensionality and data type. Points are often treated similarly to vectors
 * in mathematical operations, but conceptually they represent locations in
 * space rather than directions.
 *
 * @tparam T The data type of the point (e.g., float, double, int).
 * @tparam Size The number of dimensions (e.g., 2D, 3D, 4D).
 * @tparam Option The memory layout option (default: RowMajor).
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
using Point = Vector<T, Size, Option>;

/**
 * @brief Defines a point with `Size` dimensions and `float` precision.
 *
 * This alias provides a Point of `float` type with a specified number of
 * dimensions. The default memory layout is row-major.
 *
 * @tparam Size The number of dimensions for the point.
 * @tparam Option The memory layout option (default: RowMajor).
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using PointNf = Point<float, Size, Option>;

/**
 * @brief Defines a point with `Size` dimensions and `double` precision.
 *
 * This alias provides a Point of `double` type with a specified number of
 * dimensions. The default memory layout is row-major.
 *
 * @tparam Size The number of dimensions for the point.
 * @tparam Option The memory layout option (default: RowMajor).
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using PointNd = Point<double, Size, Option>;

/**
 * @brief Defines a point with `Size` dimensions and `int32_t` precision.
 *
 * This alias provides a Point of `int32_t` type with a specified number of
 * dimensions. The default memory layout is row-major.
 *
 * @tparam Size The number of dimensions for the point.
 * @tparam Option The memory layout option (default: RowMajor).
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using PointNi = Point<std::int32_t, Size, Option>;

/**
 * @brief Defines a point with `Size` dimensions and `uint32_t` precision.
 *
 * This alias provides a Point of `uint32_t` type with a specified number of
 * dimensions. The default memory layout is row-major.
 *
 * @tparam Size The number of dimensions for the point.
 * @tparam Option The memory layout option (default: RowMajor).
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using PointNui = Point<std::uint32_t, Size, Option>;

/**
 * @brief Templated 2D point with configurable data type.
 *
 * This alias defines a 2D point with a configurable data type.
 * The default memory layout is row-major.
 *
 * @tparam T The data type (e.g., float, double).
 * @tparam Option The memory layout option (default: RowMajor).
 */
template <typename T, Options Option = Options::RowMajor>
using Point2D = Point<T, 2, Option>;

/**
 * @brief Templated 3D point with configurable data type.
 *
 * This alias defines a 3D point with a configurable data type.
 * The default memory layout is row-major.
 *
 * @tparam T The data type (e.g., float, double).
 * @tparam Option The memory layout option (default: RowMajor).
 */
template <typename T, Options Option = Options::RowMajor>
using Point3D = Point<T, 3, Option>;

/**
 * @brief Templated 4D point with configurable data type.
 *
 * This alias defines a 4D point with a configurable data type.
 * The default memory layout is row-major.
 *
 * @tparam T The data type (e.g., float, double).
 * @tparam Option The memory layout option (default: RowMajor).
 */
template <typename T, Options Option = Options::RowMajor>
using Point4D = Point<T, 4, Option>;

// Specific data type points

/**
 * @brief 2D point of floats.
 */
using Point2Df = Point2D<float>;

/**
 * @brief 3D point of floats.
 */
using Point3Df = Point3D<float>;

/**
 * @brief 4D point of floats.
 */
using Point4Df = Point4D<float>;

/**
 * @brief 2D point of doubles.
 */
using Point2Dd = Point2D<double>;

/**
 * @brief 3D point of doubles.
 */
using Point3Dd = Point3D<double>;

/**
 * @brief 4D point of doubles.
 */
using Point4Dd = Point4D<double>;

/**
 * @brief 2D point of 32-bit integers.
 */
using Point2Di = Point2D<std::int32_t>;

/**
 * @brief 3D point of 32-bit integers.
 */
using Point3Di = Point3D<std::int32_t>;

/**
 * @brief 4D point of 32-bit integers.
 */
using Point4Di = Point4D<std::int32_t>;

/**
 * @brief 2D point of unsigned 32-bit integers.
 */
using Point2Dui = Point2D<std::uint32_t>;

/**
 * @brief 3D point of unsigned 32-bit integers.
 */
using Point3Dui = Point3D<std::uint32_t>;

/**
 * @brief 4D point of unsigned 32-bit integers.
 */
using Point4Dui = Point4D<std::uint32_t>;


}  // namespace math

#endif  // MATH_LIBRARY_POINT_H