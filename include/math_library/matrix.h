/**
 * @file matrix.h
 */

#ifndef MATH_LIBRARY_MATRIX_H
#define MATH_LIBRARY_MATRIX_H

#include "../../src/lib/options/options.h"
#include "../../src/lib/simd/instruction_set/instruction_set.h"
#include "../../src/lib/utils/concepts.h"
#include "../../src/lib/utils/utils.h"

#include <math.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <ranges>
#include <type_traits>

namespace math {

constexpr std::size_t g_kStackAllocationLimit = 16;  // 4 by 4 matrix

template <typename T, std::size_t Size, Options Option>
class Vector;

template <typename T,
          std::size_t Rows,
          std::size_t Columns,
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
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }
  }

  Matrix(const T& element) {
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }
    std::fill_n(m_data_, Rows * Columns, element);
  }

  Matrix(const Matrix& other) {
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }

    std::copy_n(other.m_data_, Rows * Columns, m_data_);
  }

  auto operator=(const Matrix& other) -> Matrix& {
    if (this != &other) {
      if constexpr (s_kUseHeap) {
        delete[] m_data_;
        m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
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
    requires AllConvertibleTo<T, Args...>
          && ArgsSizeGreaterThanCount<1, Args...>
  Matrix(Args... args) {
    static_assert(
        sizeof...(Args) == static_cast<std::size_t>(Rows) * Columns,
        "Incorrect number of arguments for Matrix initialization");
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }
    T arr[] = {[&args] { return static_cast<T>(args); }()...};
    std::copy(std::begin(arr), std::end(arr), m_data_);
  }

  template <std::input_iterator InputIt>
  Matrix(InputIt first, InputIt last) {
    assert(std::distance(first, last) == Rows * Columns);
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
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
    constexpr std::size_t kMin = std::min(Rows, Columns);
    for (std::size_t i = 0; i < kMin; ++i) {
      m(i, i) = 1;
    }
    return m;
  }

  ~Matrix() {
    if constexpr (s_kUseHeap) {
      delete[] m_data_;
    }
  }

  auto operator()(std::size_t row, std::size_t col) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_data_[row * Columns + col];
    } else {
      return m_data_[col * Rows + row];
    }
  }

  auto operator()(std::size_t row, std::size_t col) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_data_[row * Columns + col];
    } else {
      return m_data_[col * Rows + row];
    }
  }

  [[nodiscard]] auto coeff(std::size_t row, std::size_t col) const -> const T& {
    assert(row < Rows && col < Columns && "Index out of bounds");
    return operator()(row, col);
  }

  auto coeffRef(std::size_t row, std::size_t col) -> T& {
    assert(row < Rows && col < Columns && "Index out of bounds");
    return operator()(row, col);
  }

  static constexpr auto GetRows() -> std::size_t { return Rows; }

  static constexpr auto GetColumns() -> std::size_t { return Columns; }

  /**
   * @brief Calculates the total size in bytes of the matrix data.
   */
  static constexpr auto GetDataSize() -> std::size_t {
    return sizeof(T) * GetRows() * GetColumns();
  }

  static constexpr auto GetOption() -> Options { return Option; }

  template <std::size_t Row>
    requires ValueLessThan<Row, Rows>
  [[nodiscard]] auto getRow() const -> Vector<T, Columns, Option> {
    Vector<T, Columns, Option> rowVector;
    if constexpr (Option == Options::RowMajor) {
      // more optimized due to layout
      std::copy(this->data() + Row * Columns,
                this->data() + (Row + 1) * Columns,
                rowVector.data());
    } else if constexpr (Option == Options::ColumnMajor) {
      for (std::size_t col = 0; col < Columns; ++col) {
        rowVector(col) = this->operator()(Row, col);
      }
    }
    return rowVector;
  }

  template <std::size_t Col>
    requires ValueLessThan<Col, Columns>
  [[nodiscard]] auto getColumn() const -> Vector<T, Rows, Option> {
    Vector<T, Rows, Option> columnVector;
    if constexpr (Option == Options::ColumnMajor) {
      // more optimized due to layout
      std::copy(this->data() + Col * Rows,
                this->data() + (Col + 1) * Rows,
                columnVector.data());
    } else if constexpr (Option == Options::RowMajor) {
      for (std::size_t row = 0; row < Rows; ++row) {
        columnVector(row) = this->operator()(row, Col);
      }
    }
    return columnVector;
  }

  template <std::size_t Row>
    requires ValueLessThan<Row, Rows>
  void setRow(const Vector<T, Columns, Option>& vector) {
    if constexpr (Option == Options::RowMajor) {
      // more optimized due to layout
      std::copy(
          vector.data(), vector.data() + Columns, this->data() + Row * Columns);
    } else if constexpr (Option == Options::ColumnMajor) {
      for (std::size_t col = 0; col < Columns; ++col) {
        operator()(Row, col) = vector(col);
      }
    }
  }

  template <std::size_t Col>
    requires ValueLessThan<Col, Columns>
  void setColumn(const Vector<T, Rows, Option>& vector) {
    if constexpr (Option == Options::ColumnMajor) {
      // more optimized due to layout
      std::copy(vector.data(), vector.data() + Rows, this->data() + Col * Rows);
    } else if constexpr (Option == Options::RowMajor) {
      for (std::size_t row = 0; row < Rows; ++row) {
        operator()(row, Col) = vector(row);
      }
    }
  }

  template <std::size_t Index>
    requires(Option == Options::RowMajor ? ValueLessThan<Index, Rows>
                                         : ValueLessThan<Index, Columns>)
  void setBasis(
      const Vector<T, (Option == Options::RowMajor ? Columns : Rows), Option>&
          vector) {
    if constexpr (Option == Options::RowMajor) {
      setRow<Index>(vector);
    } else if constexpr (Option == Options::ColumnMajor) {
      setColumn<Index>(vector);
    }
  }

  void setBasisX(
      const Vector<T, (Option == Options::RowMajor ? Columns : Rows), Option>&
          vector)
    requires(Option == Options::RowMajor ? ValueLessThan<0, Columns>
                                         : ValueLessThan<0, Rows>)
  {
    setBasis<0>(vector);
  }

  void setBasisY(
      const Vector<T, (Option == Options::RowMajor ? Columns : Rows), Option>&
          vector)
    requires(Option == Options::RowMajor ? ValueLessThan<1, Columns>
                                         : ValueLessThan<1, Rows>)
  {
    setBasis<1>(vector);
  }

  void setBasisZ(
      const Vector<T, (Option == Options::RowMajor ? Columns : Rows), Option>&
          vector)
    requires(Option == Options::RowMajor ? ValueLessThan<2, Columns>
                                         : ValueLessThan<2, Rows>)
  {
    setBasis<2>(vector);
  }

  auto data() -> T* { return m_data_; }

  [[nodiscard]] auto data() const -> const T* { return m_data_; }

  [[nodiscard]] auto transpose() const -> Matrix<T, Columns, Rows, Option> {
    Matrix<T, Columns, Rows, Option> res;
    for (std::size_t i = 0; i < Rows; ++i) {
      for (std::size_t j = 0; j < Columns; ++j) {
        res(j, i) = (*this)(i, j);
      }
    }
    return res;
  }

  // TODO: consider for determinant method return type similar to T but with
  // trait for mapping unsigned types to signed types + std::conditional_t

  template <typename ReturnType = float>
  [[nodiscard]] auto determinant() const -> ReturnType {
    static_assert(Rows == Columns,
                  "Determinant is only defined for square matrices");
    assert(Rows == Columns);

    if constexpr (Rows == 1) {
      return static_cast<ReturnType>(m_data_[0]);
    } else if constexpr (Rows == 2) {
      const auto& a = static_cast<ReturnType>(operator()(0, 0));
      const auto& b = static_cast<ReturnType>(operator()(0, 1));
      const auto& c = static_cast<ReturnType>(operator()(1, 0));
      const auto& d = static_cast<ReturnType>(operator()(1, 1));
      return a * d - b * c;
    } else {
      ReturnType det  = 0;
      int        sign = 1;
      for (std::size_t i = 0; i < Rows; ++i) {
        Matrix<std::remove_cv_t<T>, Rows - 1, Columns - 1, Option> submatrix;
        for (std::size_t j = 1; j < Rows; ++j) {
          std::size_t k = 0;
          for (std::size_t l = 0; l < Columns; ++l) {
            if (l != i) {
              submatrix(j - 1, k) = (*this)(j, l);
              ++k;
            }
          }
        }
        // Recursive call
        det += sign * static_cast<ReturnType>((*this)(0, i))
             * submatrix.template determinant<ReturnType>();
        sign = -sign;
      }
      return det;
    }
  }

#ifdef GAUSS_JORDAN_ELIMINATION_MATRIX_INVERSE

  [[nodiscard]] auto inverse() const -> Matrix {
    static_assert(Rows == Columns,
                  "Inverse is only defined for square matrices");

    Matrix<T, Rows, 2 * Columns, Option> augmentedMatrix;
    // Fill augmentedMatrix
    for (std::size_t i = 0; i < Rows; ++i) {
      for (std::size_t j = 0; j < Columns; ++j) {
        augmentedMatrix(i, j) = (*this)(i, j);
      }
      for (std::size_t j = Columns; j < 2 * Columns; ++j) {
        if (i == j - Columns) {
          augmentedMatrix(i, j) = 1;
        } else {
          augmentedMatrix(i, j) = 0;
        }
      }
    }

    // Perform Gauss-Jordan elimination
    for (std::size_t i = 0; i < Rows; ++i) {
      // Search for maximum in this column
      T           maxEl  = math::g_abs(augmentedMatrix(i, i));
      std::size_t maxRow = i;
      for (std::size_t k = i + 1; k < Rows; ++k) {
        if (math::g_abs(augmentedMatrix(k, i)) > maxEl) {
          maxEl  = augmentedMatrix(k, i);
          maxRow = k;
        }
      }

      // Swap maximum row with current row
      for (std::size_t k = i; k < 2 * Columns; ++k) {
        T tmp                      = augmentedMatrix(maxRow, k);
        augmentedMatrix(maxRow, k) = augmentedMatrix(i, k);
        augmentedMatrix(i, k)      = tmp;
      }

      // Make all Rows below this one 0 in current column
      for (std::size_t k = i + 1; k < Rows; ++k) {
        T c = -augmentedMatrix(k, i) / augmentedMatrix(i, i);
        for (std::size_t j = i; j < 2 * Columns; ++j) {
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
        for (std::size_t j = i; j < 2 * Columns; ++j) {
          if (i == j) {
            augmentedMatrix(k, j) = 0;
          } else {
            augmentedMatrix(k, j) += c * augmentedMatrix(i, j);
          }
        }
      }
    }

    // Normalize diagonal
    for (std::size_t i = 0; i < Rows; ++i) {
      T c = 1.0 / augmentedMatrix(i, i);
      for (std::size_t j = i; j < 2 * Columns; ++j) {
        augmentedMatrix(i, j) *= c;
      }
    }

    // Copy the right half of the augmented matrix to the result
    Matrix<T, Rows, Columns, Option> inverseMatrix;
    for (std::size_t i = 0; i < Rows; ++i) {
      for (std::size_t j = 0; j < Columns; ++j) {
        inverseMatrix(i, j) = augmentedMatrix(i, j + Columns);
      }
    }

    return inverseMatrix;
  }

#endif  // GAUSS_JORDAN_ELIMINATION_MATRIX_INVERSE

#ifdef LU_DECOMPOSITION_MATRIX_INVERSE


#endif  // LU_DECOMPOSITION_MATRIX_INVERSE

  [[nodiscard]] auto rank() const -> std::size_t {
    // Create a copy of the matrix
    // TODO: using float matrix - temp solution
    Matrix<float, Rows, Columns, Option> copy;
    // TODO: consider moving convert type logic to separate method
    std::transform(
        this->data(),
        this->data() + Rows * Columns,
        copy.data(),
        [](const T& element) -> float { return static_cast<float>(element); });

    // Apply Gaussian elimination
    std::size_t rank = 0;
    for (std::size_t row = 0; row < Rows; ++row) {
      // Find the maximum element in this column
      auto        maxEl  = math::g_abs(copy(row, rank));
      std::size_t maxRow = row;
      for (std::size_t i = row + 1; i < Rows; ++i) {
        if (math::g_abs(copy(i, rank)) > maxEl) {
          maxEl  = math::g_abs(copy(i, rank));
          maxRow = i;
        }
      }

      // Swap maximum row with current row
      if (maxEl != 0) {
        for (std::size_t i = 0; i < Columns; ++i) {
          auto tmp        = copy(maxRow, i);
          copy(maxRow, i) = copy(row, i);
          copy(row, i)    = tmp;
        }

        // Make all Rows below this one 0 in current column
        for (std::size_t i = row + 1; i < Rows; ++i) {
          auto c = -copy(i, rank) / copy(row, rank);
          for (std::size_t j = rank; j < Columns; ++j) {
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
  [[nodiscard]] auto magnitude() const -> T
    requires OneDimensional<Rows, Columns>
  {
    T                      sum              = 0;
    constexpr std::size_t kVectorDimention = 1;
    constexpr std::size_t kMatrixSize      = Rows * Columns;
    auto mulFunc = InstructionSet<T>::template GetMulFunc<Option>();
    mulFunc(&sum,
            m_data_,
            m_data_,
            kVectorDimention,
            kVectorDimention,
            kMatrixSize);
    return std::sqrt(sum);
  }

  /**
   * \brief Calculates the squared Frobenius norm (squared magnitude) of a
   * matrix.
   */
  [[nodiscard]] auto magnitudeSquared() const -> T
    requires OneDimensional<Rows, Columns>
  {
    T                     result           = 0;
    constexpr std::size_t kVectorDimention = 1;
    constexpr std::size_t kMatrixSize      = Rows * Columns;
    auto mulFunc = InstructionSet<T>::template GetMulFunc<Option>();
    mulFunc(&result,
            m_data_,
            m_data_,
            kVectorDimention,
            kVectorDimention,
            kMatrixSize);
    return result;
  }

  /**
   * @brief Normalizes the matrix based on its Frobenius norm (in-place).
   *
   * @note This method modifies the matrix itself.
   */
  void normalize()
    requires OneDimensional<Rows, Columns>
  {
    T mag = magnitude();
    assert(mag != 0 && "Normalization error: magnitude is zero, implying a zero matrix/vector");
    *this /= mag;
  }

  /**
   * @brief Normalizes the matrix based on its Frobenius norm (non-in-place).
   *
   * @return A new normalized matrix.
   */
  [[nodiscard]] auto normalized() const -> Matrix
    requires OneDimensional<Rows, Columns>
  {
    T mag = magnitude();
    assert(mag != 0 && "Normalization error: magnitude is zero, implying a zero matrix/vector");
    return *this / mag;
  }

  template <std::size_t OtherRows, std::size_t OtherColumns>
    requires OneDimensional<Rows, Columns>
          && OneDimensional<OtherRows, OtherColumns>
          && SameSize<Matrix<T, Rows, Columns>,
                      Matrix<T, OtherRows, OtherColumns>>
  [[nodiscard]] auto dot(const Matrix<T, OtherRows, OtherColumns>& other) const
      -> T {
    T                      result           = NAN;
    constexpr std::size_t kVectorDimention = 1;
    constexpr std::size_t kMatrixSize      = Rows * Columns;
    auto mulFunc = InstructionSet<T>::template GetMulFunc<Option>();
    mulFunc(&result,
            m_data_,
            other.data(),
            kVectorDimention,
            kVectorDimention,
            kMatrixSize);
    return result;
  }

  template <std::size_t OtherRows, std::size_t OtherColumns>
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
    for (std::size_t i = 0; i < Rows; ++i) {
      sum += this->operator()(i, i);
    }
    return sum;
  }

  template <std::size_t NewRows, std::size_t NewColumns>
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

  auto operator-() const -> Matrix {
    Matrix result  = *this;
    auto   negFunc = InstructionSet<T>::GetNegFunc();
    negFunc(result.data(), Rows * Columns);
    return result;
  }

  template <std::size_t ResultColumns>
  auto operator*(const Matrix<T, Columns, ResultColumns, Option>& other) const
      -> Matrix<T, Rows, ResultColumns, Option> {
    Matrix<T, Rows, ResultColumns, Option> result;
    auto mulFunc = InstructionSet<T>::template GetMulFunc<Option>();
    mulFunc(result.data(), m_data_, other.data(), Rows, ResultColumns, Columns);
    return result;
  }

  /**
   * @brief Multiplies this matrix with another matrix and updates this matrix
   * with the result.
   *
   * @note 'Columns' is used for both dimensions to reflect the nature of square
   * matrices and to make sure that the result matrix will be the same dimention
   * as original one.
   */
  template <std::size_t OtherRows, std::size_t OtherColumns>
  auto operator*=(const Matrix<T, OtherRows, OtherColumns, Option>& other)
      -> Matrix&
    requires SquaredMatrix<Matrix<T, OtherRows, OtherColumns, Option>>
  {
    if constexpr (Option == Options::RowMajor) {
      *this = *this * other;
    } else {
      *this = other * (*this);
    }
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

  auto operator==(const Matrix& other) const -> bool {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
      // floating point types
      constexpr T kEpsilon = std::numeric_limits<T>::epsilon();
      auto almostEqual = [](T a, T b) { return std::fabs(a - b) <= kEpsilon; };
      return std::equal(
          m_data_, m_data_ + Rows * Columns, other.m_data_, almostEqual);
    } else {
      // general types
      return std::equal(m_data_, m_data_ + Rows * Columns, other.m_data_);
    }
  }

  auto operator!=(const Matrix& other) const -> bool {
    return !(*this == other);
  }

  friend auto operator<<(std::ostream& os, const Matrix& matrix)
      -> std::ostream& {
    for (std::size_t i = 0; i < Rows; ++i) {
      os << matrix(i, 0);
      for (std::size_t j = 1; j < Columns; ++j) {
        os << ' ' << matrix(i, j);
      }
      if (i < Rows - 1) {
        os << '\n';
      }
    }
    return os;
  }

  class MatrixInitializer {
    Matrix&     m_matrix_;
    std::size_t m_row_;
    std::size_t m_col_;

    public:
    MatrixInitializer(Matrix& matrix)
        : m_matrix_(matrix)
        , m_row_(0)
        , m_col_(0) {}

    auto operator,(const T& value) -> MatrixInitializer& {
      m_matrix_.operator()(m_row_, m_col_) = value;

      ++m_col_;
      if (m_col_ == Columns) {
        m_col_ = 0;
        ++m_row_;
      }

      return *this;
    }
  };

  /**
   * @brief overload operator << for initializing Matrix elements in an
   * intuitive manner using the comma operator.
   *
   * Usage example for a row-major matrix:
   * @code
   * math::MatrixNf<2, 3, math::Options::RowMajor> m1;
   * m1 << 1, 2, 3,
   *       4, 5, 6;
   * // Output : 1, 2, 3,
   *             4, 5, 6
   * @endcode
   *
   * Usage example for a column-major matrix:
   * @code
   * math::MatrixNf<3, 2, math::Options::ColumnMajor> m2;
   * m2 << 1, 4,
   *       2, 5,
   *       3, 6;
   * // Output : 1, 2,
   *             3, 4,
   *             5, 6,
   * @endcode
   *
   * @note Initialization through the << operator does not assign elements based
   * on how they will be stored in memory, as Matrix(Args...) does.
   */
  auto operator<<(const T& value) -> MatrixInitializer {
    // Resetting the matrix initializer with every new << operation
    MatrixInitializer initializer(*this);
    initializer, value;  // Start the chain of insertions
    return initializer;
  }
};

/**
 * @brief Performs scalar multiplication on a Matrix, where the scalar value is
 * the left-hand operand. (scalar * Matrix)
 */
template <typename ScalarType,
          typename T,
          std::size_t Rows,
          std::size_t Columns,
          Options     Option>
auto operator*(const ScalarType&                       scalar,
               const Matrix<T, Rows, Columns, Option>& matrix)
    -> Matrix<T, Rows, Columns, Option> {
  return matrix * static_cast<T>(scalar);
}

// Matrix of floats
template <std::size_t Rows,
          std::size_t Columns,
          Options     Option = Options::RowMajor>
using MatrixNf = Matrix<float, Rows, Columns, Option>;

// Matrix of doubles
template <std::size_t Rows,
          std::size_t Columns,
          Options     Option = Options::RowMajor>
using MatrixNd = Matrix<double, Rows, Columns, Option>;

// Matrix of ints
template <std::size_t Rows,
          std::size_t Columns,
          Options      Option = Options::RowMajor>
using MatrixNi = Matrix<std::int32_t, Rows, Columns, Option>;

// TODO: add Option template parameter to the alias below (may introduce the
// problem with adding <> to the end of the alias - e.g. Matrix2f<> for default
// value)

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
