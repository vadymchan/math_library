/**
 * @file concepts.h
 * @brief This file defines various concepts used in the math library.
 *
 * The concepts defined in this file are used to enforce certain properties
 * in template-based code, improving compile-time checks and ensuring correct
 * usage of mathematical constructs such as matrices, vectors, and general
 * parameters.
 */

#ifndef MATH_LIBRARY_CONCEPTS
#define MATH_LIBRARY_CONCEPTS

#include <concepts>

namespace math {

/**
 * @brief Ensures that a value is at least a minimum value.
 *
 * This concept checks that the provided value is greater than or equal to the
 * specified minimum value.
 *
 * @tparam Value The value to be checked.
 * @tparam MinValue The minimum allowed value.
 */
template <auto Value, auto MinValue>
concept ValueAtLeast = Value >= MinValue;

/**
 * @brief Ensures that a value is less than a given limit.
 *
 * This concept checks that the provided value is less than the specified limit.
 *
 * @tparam Value The value to be checked.
 * @tparam Limit The upper limit for the value.
 */
template <auto Value, auto Limit>
concept ValueLessThan = Value < Limit;

/**
 * @brief Ensures that a value is equal to a specific value.
 *
 * This concept checks that the provided value is exactly equal to the specified
 * exact value.
 *
 * @tparam Value The value to be checked.
 * @tparam ExactValue The value that the input should match.
 */
template <auto Value, auto ExactValue>
concept ValueEqualTo = Value == ExactValue;

/**
 * @brief Checks if a matrix is one-dimensional.
 *
 * This concept checks whether the matrix is one-dimensional, i.e., has either
 * one row or one column.
 *
 * @tparam Rows Number of rows in the matrix.
 * @tparam Columns Number of columns in the matrix.
 */
template <std::size_t Rows, std::size_t Columns>
concept OneDimensional = (Rows == 1 || Columns == 1);

/**
 * @brief Ensures that all arguments are convertible to a specified type.
 *
 * This concept checks if all arguments in the variadic template can be
 * converted to a given type.
 *
 * @tparam U The target type to which all arguments should be convertible.
 * @tparam Args The types of the arguments to be checked.
 */
template <typename U, typename... Args>
concept AllConvertibleTo = (std::convertible_to<Args, U> && ...);

/**
 * @brief Ensures that the number of arguments is greater than a specified
 * count.
 *
 * This concept checks whether the number of arguments in the variadic template
 * exceeds the given count.
 *
 * @tparam Count The minimum number of arguments.
 * @tparam Args The types of the arguments to be checked.
 */
template <auto Count, typename... Args>
concept ArgsSizeGreaterThanCount = (sizeof...(Args) > Count);

/**
 * @brief Ensures that a matrix represents a three-dimensional vector.
 *
 * This concept checks whether the matrix has three rows and one column, or one
 * row and three columns.
 *
 * @tparam MatrixType The type of the matrix being checked.
 */
template <typename MatrixType>
concept ThreeDimensionalVector
    = ((MatrixType::GetRows() == 3 && MatrixType::GetColumns() == 1)
       || (MatrixType::GetRows() == 1 && MatrixType::GetColumns() == 3));

/**
 * @brief Ensures that two matrices have the same size.
 *
 * This concept checks that the two matrices have the same number of elements.
 *
 * @tparam MatrixA The first matrix type.
 * @tparam MatrixB The second matrix type.
 */
template <typename MatrixA, typename MatrixB>
concept SameSize = (MatrixA::GetRows() * MatrixA::GetColumns()
                    == MatrixB::GetRows() * MatrixB::GetColumns());

/**
 * @brief Checks if a matrix is square.
 *
 * This concept checks whether a matrix has the same number of rows and columns.
 *
 * @tparam Matrix The matrix type to be checked.
 */
template <typename Matrix>
concept SquaredMatrix = (Matrix::GetRows() == Matrix::GetColumns());

}  // namespace math

#endif  // MATH_LIBRARY_CONCEPTS
