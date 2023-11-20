#ifndef MATH_LIBRARY_CONCEPTS
#define MATH_LIBRARY_CONCEPTS

#include <concepts>

namespace math {

template <auto Value, auto ExactValue>
concept ValueEqualTo = Value == ExactValue;

template <typename U, typename... Args>
concept AllSameAs = (std::same_as<Args, U> && ...);

template <auto Count, typename... Args>
concept ArgsSizeGreaterThanCount = (sizeof...(Args) > Count);

template <typename MatrixType>
concept OneDimensional
    = (MatrixType::GetRows() == 1 || MatrixType::GetColumns() == 1);

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

#endif  // !MATH_LIBRARY_CONCEPTS
