/**
 * @file concepts.h
 */

#ifndef MATH_LIBRARY_CONCEPTS
#define MATH_LIBRARY_CONCEPTS

#include <concepts>

namespace math {

template <auto Value, auto MinValue>
concept ValueAtLeast = Value >= MinValue;

template <auto Value, auto Limit>
concept ValueLessThan = Value < Limit;

template <auto Value, auto ExactValue>
concept ValueEqualTo = Value == ExactValue;

template <unsigned int Rows, unsigned int Columns>
concept OneDimensional = (Rows == 1 || Columns == 1);

template <typename U, typename... Args>
concept AllConvertibleTo = (std::convertible_to<Args, U> && ...);

template <auto Count, typename... Args>
concept ArgsSizeGreaterThanCount = (sizeof...(Args) > Count);

template <typename MatrixType>
concept ThreeDimensionalVector
    = ((MatrixType::GetRows() == 3 && MatrixType::GetColumns() == 1)
       || (MatrixType::GetRows() == 1 && MatrixType::GetColumns() == 3));

template <typename MatrixA, typename MatrixB>
concept SameSize = (MatrixA::GetRows() * MatrixA::GetColumns()
                    == MatrixB::GetRows() * MatrixB::GetColumns());

template <typename Matrix>
concept SquaredMatrix = (Matrix::GetRows() == Matrix::GetColumns());

}  // namespace math

#endif
