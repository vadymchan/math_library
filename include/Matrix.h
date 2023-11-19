/**
 * @file Matrix.h
 */

#ifndef MATH_LIBRARY_MATRIX_H
#define MATH_LIBRARY_MATRIX_H

#include "../src/lib/Options/Options.h"
#include "../src/lib/simd/instruction_set/InstructionSet.h"
#include "../src/lib/utils/Concepts.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <ranges>
#include <type_traits>

constexpr unsigned int g_kStackAllocationLimit = 16;  // 4 by 4 matrix

namespace math {

template <typename T,
          unsigned int Rows,
          unsigned int Columns,
          Options      Option = Options::RowMajor>
class Matrix {
  public:
  static const bool s_kUseHeap = Rows * Columns > g_kStackAllocationLimit;

  private:
  using DataType = std::conditional_t<s_kUseHeap, T*, T[Rows * Columns]>;
  DataType m_data_;

  public:
  Matrix() {
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<unsigned long long>(Rows) * Columns];
    }
  }

  Matrix(const T& element) {
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<unsigned long long>(Rows) * Columns];
    }
    std::fill_n(m_data_, Rows * Columns, element);
  }

  Matrix(const Matrix& other) {
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<unsigned long long>(Rows) * Columns];
    }

    std::copy_n(other.m_data_, Rows * Columns, m_data_);
  }

  auto operator=(const Matrix& other) -> Matrix& {
    if (this != &other) {
      if constexpr (s_kUseHeap) {
        delete[] m_data_;
        m_data_ = new T[static_cast<unsigned long long>(Rows) * Columns];
      }
      std::copy_n(other.m_data_, Rows * Columns, m_data_);
    }
    return *this;
  }

  Matrix(Matrix&& other) noexcept {
    if constexpr (s_kUseHeap) {
      m_data_       = other.m_data_;
      other.m_data_ = nullptr;
    } else {
      std::move(std::begin(other.m_data_),
                std::end(other.m_data_),
                std::begin(m_data_));
    }
  }

  auto operator=(Matrix&& other) noexcept -> Matrix& {
    if (this != &other) {
      if constexpr (s_kUseHeap) {
        delete[] m_data_;
        m_data_       = other.m_data_;
        other.m_data_ = nullptr;
      } else {
        std::move(std::begin(other.m_data_),
                  std::end(other.m_data_),
                  std::begin(m_data_));
      }
    }
    return *this;
  }

  template <typename... Args>
    requires AllSameAs<T, Args...> && ArgsSizeGreaterThanCount<1, Args...>
  Matrix(Args... args) {
    static_assert(
        sizeof...(Args) == static_cast<unsigned long long>(Rows) * Columns,
        "Incorrect number of arguments for Matrix initialization");
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<unsigned long long>(Rows) * Columns];
    }
    T arr[] = {args...};
    std::copy(std::begin(arr), std::end(arr), m_data_);
  }

  template <std::input_iterator InputIt>
  Matrix(InputIt first, InputIt last) {
    assert(std::distance(first, last) == Rows * Columns);
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<unsigned long long>(Rows) * Columns];
    }
    std::copy(first, last, m_data_);
  }

  template <std::ranges::range Range>
  Matrix(const Range& range) {
    assert(std::ranges::size(range) <= Rows * Columns);
    if constexpr (s_kUseHeap) {
      m_data_ = new T[Rows * Columns];
    }
    std::copy_n(range.begin(), Rows * Columns, m_data_);
  }

  static constexpr auto Identity() -> Matrix {
    Matrix                 m(0);
    constexpr unsigned int kMin = std::min(Rows, Columns);
    for (unsigned int i = 0; i < kMin; ++i) {
      m(i, i) = 1;
    }
    return m;
  }

  ~Matrix() {
    if constexpr (s_kUseHeap) {
      delete[] m_data_;
    }
  }

  auto operator()(unsigned int row, unsigned int col) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_data_[row * Columns + col];
    } else {
      return m_data_[col * Rows + row];
    }
  }

  auto operator()(unsigned int row, unsigned int col) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_data_[row * Columns + col];
    } else {
      return m_data_[col * Rows + row];
    }
  }

  [[nodiscard]] auto coeff(unsigned int row, unsigned int col) const
      -> const T& {
    assert(row < Rows && col < Columns && "Index out of bounds");
    return operator()(row, col);
  }

  auto coeffRef(unsigned int row, unsigned int col) -> T& {
    assert(row < Rows && col < Columns && "Index out of bounds");
    return operator()(row, col);
  }

  static constexpr auto GetRows() -> unsigned int { return Rows; } 

  static constexpr auto GetColumns() -> unsigned int { return Columns; }

  static constexpr auto GetOption() -> Options { return Option; }

  auto data() -> T* { return m_data_; }

  [[nodiscard]] auto data() const -> const T* { return m_data_; }

  [[nodiscard]] auto transpose() const -> Matrix<T, Columns, Rows, Option> {
    Matrix<T, Columns, Rows, Option> res;
    for (unsigned int i = 0; i < Rows; ++i) {
      for (unsigned int j = 0; j < Columns; ++j) {
        res(j, i) = (*this)(i, j);
      }
    }
    return res;
  }

  [[nodiscard]] auto determinant() const -> T {
    static_assert(Rows == Columns,
                  "Determinant is only defined for square matrices");
    assert(Rows == Columns);

    if constexpr (Rows == 1) {
      return m_data_[0];
    } else if constexpr (Rows == 2) {
      const T& a = operator()(0, 0);
      const T& b = operator()(0, 1);
      const T& c = operator()(1, 0);
      const T& d = operator()(1, 1);
      return a * d - b * c;
    } else {
      T   det  = 0;
      int sign = 1;
      for (unsigned int i = 0; i < Rows; ++i) {
        // Construct a sub-matrix
        Matrix<T, Rows - 1, Columns - 1, Option> submatrix;
        for (unsigned int j = 1; j < Rows; ++j) {
          unsigned int k = 0;
          for (unsigned int l = 0; l < Columns; ++l) {
            if (l != i) {
              submatrix(j - 1, k) = (*this)(j, l);
              ++k;
            }
          }
        }
        // Recursive call
        det  += sign * (*this)(0, i) * submatrix.determinant();
        sign  = -sign;
      }
      return det;
    }
  }

  [[nodiscard]] auto inverse() const -> Matrix {
    static_assert(Rows == Columns,
                  "Inverse is only defined for square matrices");

    Matrix<T, Rows, 2 * Columns, Option> augmentedMatrix;
    // Fill augmentedMatrix
    for (unsigned int i = 0; i < Rows; ++i) {
      for (unsigned int j = 0; j < Columns; ++j) {
        augmentedMatrix(i, j) = (*this)(i, j);
      }
      for (unsigned int j = Columns; j < 2 * Columns; ++j) {
        if (i == j - Columns) {
          augmentedMatrix(i, j) = 1;
        } else {
          augmentedMatrix(i, j) = 0;
        }
      }
    }

    // Perform Gauss-Jordan elimination
    for (unsigned int i = 0; i < Rows; ++i) {
      // Search for maximum in this column
      T            maxEl  = std::abs(augmentedMatrix(i, i));
      unsigned int maxRow = i;
      for (unsigned int k = i + 1; k < Rows; ++k) {
        if (std::abs(augmentedMatrix(k, i)) > maxEl) {
          maxEl  = augmentedMatrix(k, i);
          maxRow = k;
        }
      }

      // Swap maximum row with current row
      for (unsigned int k = i; k < 2 * Columns; ++k) {
        T tmp                      = augmentedMatrix(maxRow, k);
        augmentedMatrix(maxRow, k) = augmentedMatrix(i, k);
        augmentedMatrix(i, k)      = tmp;
      }

      // Make all Rows below this one 0 in current column
      for (unsigned int k = i + 1; k < Rows; ++k) {
        T c = -augmentedMatrix(k, i) / augmentedMatrix(i, i);
        for (unsigned int j = i; j < 2 * Columns; ++j) {
          if (i == j) {
            augmentedMatrix(k, j) = 0;
          } else {
            augmentedMatrix(k, j) += c * augmentedMatrix(i, j);
          }
        }
      }
    }

    // Make all Rows above this one zero in current column
    for (int i = Rows - 1; i >= 0; i--) {
      for (int k = i - 1; k >= 0; k--) {
        T c = -augmentedMatrix(k, i) / augmentedMatrix(i, i);
        for (unsigned int j = i; j < 2 * Columns; ++j) {
          if (i == j) {
            augmentedMatrix(k, j) = 0;
          } else {
            augmentedMatrix(k, j) += c * augmentedMatrix(i, j);
          }
        }
      }
    }

    // Normalize diagonal
    for (unsigned int i = 0; i < Rows; ++i) {
      T c = 1.0 / augmentedMatrix(i, i);
      for (unsigned int j = i; j < 2 * Columns; ++j) {
        augmentedMatrix(i, j) *= c;
      }
    }

    // Copy the right half of the augmented matrix to the result
    Matrix<T, Rows, Columns, Option> inverseMatrix;
    for (unsigned int i = 0; i < Rows; ++i) {
      for (unsigned int j = 0; j < Columns; ++j) {
        inverseMatrix(i, j) = augmentedMatrix(i, j + Columns);
      }
    }

    return inverseMatrix;
  }

  [[nodiscard]] auto rank() const -> int {
    // Create a copy of the matrix
    Matrix<T, Rows, Columns, Option> copy(*this);

    // Apply Gaussian elimination
    int rank = 0;
    for (int row = 0; row < Rows; ++row) {
      // Find the maximum element in this column
      T   maxEl  = std::abs(copy(row, rank));
      int maxRow = row;
      for (int i = row + 1; i < Rows; ++i) {
        if (std::abs(copy(i, rank)) > maxEl) {
          maxEl  = std::abs(copy(i, rank));
          maxRow = i;
        }
      }

      // Swap maximum row with current row
      if (maxEl != 0) {
        for (int i = 0; i < Columns; ++i) {
          T tmp           = copy(maxRow, i);
          copy(maxRow, i) = copy(row, i);
          copy(row, i)    = tmp;
        }

        // Make all Rows below this one 0 in current column
        for (int i = row + 1; i < Rows; ++i) {
          T c = -copy(i, rank) / copy(row, rank);
          for (int j = rank; j < Columns; ++j) {
            if (rank == j) {
              copy(i, j) = 0;
            } else {
              copy(i, j) += c * copy(row, j);
            }
          }
        }

        ++rank;
      }

      // If rank is equal to Columns, no need to continue
      if (rank == Columns) {
        break;
      }
    }

    return rank;
  }

  /**
   * \brief Calculates the Frobenius norm (magnitude) of a matrix.
   */
  [[nodiscard]] auto magnitude() const -> T {
    T                      sum              = 0;
    constexpr unsigned int kVectorDimention = 1;
    constexpr unsigned int kMatrixSize      = Rows * Columns;
    auto mulFunc = InstructionSet<T>::template GetMulFunc<Option>();
    mulFunc(&sum,
            m_data_,
            m_data_,
            kVectorDimention,
            kVectorDimention,
            kMatrixSize);
    return std::sqrt(sum);
  }

#ifdef USE_NORMALIZE_IN_PLACE
  /**
   * \brief Normalizes the matrix based on its Frobenius norm.
   */
  void normalize() {
    T mag = magnitude();
    assert(mag != 0 && "Normalization error: magnitude is zero, implying a zero matrix/vector");
    *this /= mag;
  }

#else
  /**
   * \brief Normalizes the matrix based on its Frobenius norm.
   */
  [[nodiscard]] auto normalize() const -> Matrix {
    T mag = magnitude();
    assert(mag != 0 && "Normalization error: magnitude is zero, implying a zero matrix/vector");

    Matrix<T, Rows, Columns, Option> result(*this);
    result /= mag;
    return result;
  }

#endif  // USE_NORMALIZE_IN_PLACE

  template <unsigned int OtherRows, unsigned int OtherColumns>
    requires OneDimensional<Matrix<T, Rows, Columns>>
          && OneDimensional<Matrix<T, OtherRows, OtherColumns>>
          && SameSize<Matrix<T, Rows, Columns>,
                      Matrix<T, OtherRows, OtherColumns>>
  [[nodiscard]] auto dot(const Matrix<T, OtherRows, OtherColumns>& other) const
      -> T {
    float                  result           = NAN;
    constexpr unsigned int kVectorDimention = 1;
    constexpr unsigned int kMatrixSize      = Rows * Columns;
    auto mulFunc = InstructionSet<T>::template GetMulFunc<Option>();
    mulFunc(&result,
            m_data_,
            other.data(),
            kVectorDimention,
            kVectorDimention,
            kMatrixSize);
    return result;
  }

  template <unsigned int OtherRows, unsigned int OtherColumns>
    requires ThreeDimensionalVector<Matrix<T, OtherRows, OtherColumns>>
          && ThreeDimensionalVector<Matrix<T, Rows, Columns>>
  [[nodiscard]] auto cross(
      const Matrix<T, OtherRows, OtherColumns>& other) const -> Matrix {
    Matrix<T, 3, 1, Option> result;

    result(0, 0) = this->operator()(1, 0) * other(2, 0)
                 - this->operator()(2, 0) * other(1, 0);
    result(1, 0) = this->operator()(2, 0) * other(0, 0)
                 - this->operator()(0, 0) * other(2, 0);
    result(2, 0) = this->operator()(0, 0) * other(1, 0)
                 - this->operator()(1, 0) * other(0, 0);

    return result;
  }

  [[nodiscard]] auto trace() const -> T {
    static_assert(Rows == Columns, "Trace is only defined for square matrices");
    T sum = 0;
    for (unsigned int i = 0; i < Rows; ++i) {
      sum += this->operator()(i, i);
    }
    return sum;
  }

  template <unsigned int NewRows, unsigned int NewColumns>
  [[nodiscard]] auto reshape() const -> Matrix<T, NewRows, NewColumns, Option> {
    static_assert(
        Rows * Columns == NewRows * NewColumns,
        "New dimensions must have the same total size as the original matrix");
    Matrix<T, NewRows, NewColumns, Option> newMatrix;
    std::copy_n(m_data_, Rows * Columns, newMatrix.data());
    return newMatrix;
  }

  auto operator+(const Matrix& other) const -> Matrix {
    Matrix result  = *this;
    auto   addFunc = InstructionSet<T>::GetAddFunc();
    addFunc(result.m_data_, other.m_data_, Rows * Columns);
    return result;
  }

  auto operator+=(const Matrix& other) -> Matrix& {
    auto addFunc = InstructionSet<T>::GetAddFunc();
    addFunc(m_data_, other.m_data_, Rows * Columns);
    return *this;
  }

  auto operator+(const T& scalar) const -> Matrix {
    Matrix result        = *this;
    auto   addScalarFunc = InstructionSet<T>::GetAddScalarFunc();
    addScalarFunc(result.m_data_, scalar, Rows * Columns);
    return result;
  }

  auto operator+=(const T& scalar) -> Matrix& {
    auto addScalarFunc = InstructionSet<T>::GetAddScalarFunc();
    addScalarFunc(m_data_, scalar, Rows * Columns);
    return *this;
  }

  auto operator-(const Matrix& other) const -> Matrix {
    Matrix result  = *this;
    auto   subFunc = InstructionSet<T>::GetSubFunc();
    subFunc(result.m_data_, other.m_data_, Rows * Columns);
    return result;
  }

  auto operator-=(const Matrix& other) -> Matrix& {
    auto subFunc = InstructionSet<T>::GetSubFunc();
    subFunc(m_data_, other.m_data_, Rows * Columns);
    return *this;
  }

  auto operator-(const T& scalar) const -> Matrix {
    Matrix result        = *this;
    auto   subScalarFunc = InstructionSet<T>::GetSubScalarFunc();
    subScalarFunc(result.m_data_, scalar, Rows * Columns);
    return result;
  }

  auto operator-=(const T& scalar) -> Matrix& {
    auto subScalarFunc = InstructionSet<T>::GetSubScalarFunc();
    subScalarFunc(m_data_, scalar, Rows * Columns);
    return *this;
  }

  template <unsigned int ResultColumns>
  auto operator*(const Matrix<T, Columns, ResultColumns, Option>& other) const
      -> Matrix<T, Rows, ResultColumns, Option> {
    Matrix<T, Rows, ResultColumns, Option> result;
    auto mulFunc = InstructionSet<T>::template GetMulFunc<Option>();
    mulFunc(result.data(), m_data_, other.data(), Rows, ResultColumns, Columns);
    return result;
  }

  /**
   * @brief Matrix multiplication-assignment operator
   *
   * This operator multiplies the current matrix with the given one.
   * Note: This function only works when the matrices have the same dimensions
   * and squared.
   *
   */
  auto operator*=(const Matrix& other) -> Matrix& {
    static_assert(
        Columns == Rows,
        "For Matrix multiplication (*=), matrix dimensions must be squared");
    assert(this != &other
           && "Cannot perform operation on the same matrix instance");
    *this = *this * other;
    return *this;
  }

  auto operator*(const T& scalar) const -> Matrix {
    Matrix result        = *this;
    auto   mulScalarFunc = InstructionSet<T>::GetMulScalarFunc();
    mulScalarFunc(result.m_data_, scalar, Rows * Columns);
    return result;
  }

  auto operator*=(const T& scalar) -> Matrix& {
    auto mulScalarFunc = InstructionSet<T>::GetMulScalarFunc();
    mulScalarFunc(m_data_, scalar, Rows * Columns);
    return *this;
  }

  auto operator/(const T& scalar) const -> Matrix {
    Matrix result        = *this;
    auto   divScalarFunc = InstructionSet<T>::GetDivScalarFunc();
    divScalarFunc(result.m_data_, scalar, Rows * Columns);
    return result;
  }

  auto operator/=(const T& scalar) -> Matrix& {
    auto divScalarFunc = InstructionSet<T>::GetDivScalarFunc();
    divScalarFunc(m_data_, scalar, Rows * Columns);
    return *this;
  }

  friend auto operator<<(std::ostream& os, const Matrix& matrix)
      -> std::ostream& {
    for (unsigned int i = 0; i < Rows; ++i) {
      for (unsigned int j = 0; j < Columns; ++j) {
        os << matrix(i, j) << ' ';
      }
      os << '\n';
    }
    return os;
  }

  class MatrixInitializer {
    Matrix&      m_matrix_;
    unsigned int m_row_;
    unsigned int m_column_;

    public:
    MatrixInitializer(Matrix& matrix, unsigned int row, unsigned int column)
        : m_matrix_(matrix)
        , m_row_(row)
        , m_column_(column) {}

    auto operator,(const T& value) -> MatrixInitializer& {
      if (m_column_ >= m_matrix_.GetColumns()) {
        ++m_row_;
        m_column_ = 0;
      }

      if (m_row_ < m_matrix_.GetRows()) {
        m_matrix_(m_row_, m_column_) = value;
        ++m_column_;
      }

      return *this;
    }
  };

  auto operator<<(const T& value) -> MatrixInitializer {
    this->operator()(0, 0) = value;
    return MatrixInitializer(*this, 0, 1);
  }
};

template <typename T, unsigned int Rows, unsigned int Columns, Options Option>
inline constexpr auto operator==(const Matrix<T, Rows, Columns, Option>& lhs,
                                 const Matrix<T, Rows, Columns, Option>& rhs)
    -> bool {
  return std::equal(lhs.data(), lhs.data() + Rows * Columns, rhs.data());
}

template <unsigned int Rows, unsigned int Columns, Options Option>
inline constexpr auto operator==(
    const Matrix<float, Rows, Columns, Option>& lhs,
    const Matrix<float, Rows, Columns, Option>& rhs) -> bool {
  constexpr float kEpsilon = std::numeric_limits<float>::epsilon();
  auto            almostEqual
      = [](float a, float b) { return std::fabs(a - b) <= kEpsilon; };
  return std::equal(
      lhs.data(), lhs.data() + Rows * Columns, rhs.data(), almostEqual);
}

template <unsigned int Rows, unsigned int Columns, Options Option>
inline constexpr auto operator==(
    const Matrix<double, Rows, Columns, Option>& lhs,
    const Matrix<double, Rows, Columns, Option>& rhs) -> bool {
  constexpr double kEpsilon = std::numeric_limits<double>::epsilon();
  auto             almostEqual
      = [](double a, double b) { return std::fabs(a - b) <= kEpsilon; };
  return std::equal(
      lhs.data(), lhs.data() + Rows * Columns, rhs.data(), almostEqual);
}

template <typename T, unsigned int Rows, unsigned int Columns, Options Option>
inline constexpr auto operator!=(const Matrix<T, Rows, Columns, Option>& lhs,
                                 const Matrix<T, Rows, Columns, Option>& rhs)
    -> bool {
  return !(lhs == rhs);
}

// Matrix of floats
template <unsigned int Rows,
          unsigned int Columns,
          Options      Option = Options::RowMajor>
using MatrixNf = Matrix<float, Rows, Columns, Option>;

// Matrix of doubles
template <unsigned int Rows,
          unsigned int Columns,
          Options      Option = Options::RowMajor>
using MatrixNd = Matrix<double, Rows, Columns, Option>;

// Matrix of ints
template <unsigned int Rows,
          unsigned int Columns,
          Options      Option = Options::RowMajor>
using MatrixNi = Matrix<int, Rows, Columns, Option>;

// Specific size matrices and vectors
using Matrix2f = MatrixNf<2, 2>;
using Matrix3f = MatrixNf<3, 3>;
using Matrix4f = MatrixNf<4, 4>;

using Matrix2d = MatrixNd<2, 2>;
using Matrix3d = MatrixNd<3, 3>;
using Matrix4d = MatrixNd<4, 4>;

using Matrix2i = MatrixNi<2, 2>;
using Matrix3i = MatrixNi<3, 3>;
using Matrix4i = MatrixNi<4, 4>;
}  // namespace math

#endif
