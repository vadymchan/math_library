/**
 * @file matrix.h
 * @brief This file contains the Matrix class template, providing support for
 * matrix operations.
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

#define LU_DECOMPOSITION_MATRIX_INVERSE

namespace math {

/**
 * @brief Defines the limit for stack allocation in terms of matrix size.
 *
 * If the number of elements in the matrix exceeds this limit, heap allocation
 * is used instead of stack allocation. This constant is set to 16, which
 * corresponds to a 4x4 matrix.
 */
constexpr std::size_t g_kStackAllocationLimit = 16;

template <typename T, std::size_t Size, Options Option>
class Vector;

/**
 * @brief A class template for a matrix with customizable dimensions and memory
 * layout.
 *
 * This template represents a matrix with a specified number of rows and
 * columns, and supports operations like addition, subtraction, multiplication,
 * transposition, and more. It can be configured to use either row-major or
 * column-major memory layout.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Rows The number of rows in the matrix.
 * @tparam Columns The number of columns in the matrix.
 * @tparam Option Specifies whether the matrix is stored in row-major or
 * column-major order (default is row-major).
 */
template <typename T,
          std::size_t Rows,
          std::size_t Columns,
          Options     Option = Options::RowMajor>
class Matrix {
  public:
  /**
   * @brief Determines whether heap allocation is used for matrix data storage.
   *
   * This static constant is `true` if the matrix size exceeds a predefined
   * stack allocation limit, in which case memory is allocated on the heap.
   * Otherwise, stack allocation is used.
   */
  static const bool s_kUseHeap = Rows * Columns > g_kStackAllocationLimit;

  private:
  /**
   * @brief The data type used for storing matrix elements.
   *
   * If the matrix size exceeds the stack allocation limit (`s_kUseHeap ==
   * true`), the data is allocated dynamically on the heap as a pointer (`T*`).
   * Otherwise, the data is stored directly on the stack as a static array
   * (`T[Rows * Columns]`).
   */
  using DataType = std::conditional_t<s_kUseHeap, T*, T[Rows * Columns]>;

  /**
   * @brief The actual matrix data.
   *
   * This member stores the matrix elements, either on the stack or heap
   * depending on the matrix size.
   */
  DataType m_data_;

  public:
  /**
   * @brief Default constructor for creating an empty matrix.
   *
   * Allocates memory on the heap if the matrix size exceeds a predefined limit,
   * otherwise it uses stack allocation.
   */
  Matrix() {
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }
  }

  /**
   * @brief Constructs a matrix filled with a single element value.
   *
   * @param element The value to fill the matrix with.
   */
  Matrix(const T& element) {
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }
    std::fill_n(m_data_, Rows * Columns, element);
  }

  /**
   * @brief Copy constructor for the Matrix.
   *
   * Creates a deep copy of the given matrix.
   *
   * @param other The matrix to copy from.
   */
  Matrix(const Matrix& other) {
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }

    std::copy_n(other.m_data_, Rows * Columns, m_data_);
  }

  /**
   * @brief Assignment operator for copying another matrix.
   *
   * @param other The matrix to copy from.
   * @return Reference to the modified matrix.
   */
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

  /**
   * @brief Move constructor for the Matrix.
   *
   * Transfers ownership of the matrix data from the given matrix to the current
   * instance.
   *
   * @param other The matrix to move from.
   */
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

  /**
   * @brief Move assignment operator for transferring ownership of another
   * matrix.
   *
   * @param other The matrix to move from.
   * @return Reference to the modified matrix.
   */
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

  /**
   * @brief Constructs a matrix with a list of elements.
   *
   * @tparam Args The types of the elements, which must be convertible to type
   * T.
   * @param args A list of elements to initialize the matrix.
   */
  template <typename... Args>
    requires AllConvertibleTo<T, Args...>
          && ArgsSizeGreaterThanCount<1, Args...>
  Matrix(Args... args) {
    static_assert(sizeof...(Args) == static_cast<std::size_t>(Rows) * Columns,
                  "Incorrect number of arguments for Matrix initialization");
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }
    T arr[] = {[&args] { return static_cast<T>(args); }()...};
    std::copy(std::begin(arr), std::end(arr), m_data_);
  }

  /**
   * @brief Constructs a matrix from a range of input iterators.
   *
   * @tparam InputIt The type of the input iterator.
   * @param first The iterator pointing to the first element.
   * @param last The iterator pointing to the last element.
   */
  template <std::input_iterator InputIt>
  Matrix(InputIt first, InputIt last) {
    assert(std::distance(first, last) == Rows * Columns);
    if constexpr (s_kUseHeap) {
      m_data_ = new T[static_cast<std::size_t>(Rows) * Columns];
    }
    std::copy(first, last, m_data_);
  }

  /**
   * @brief Constructs a matrix from a range.
   *
   * @tparam Range The type of the range.
   * @param range The range to copy the elements from.
   */
  template <std::ranges::range Range>
  Matrix(const Range& range) {
    assert(std::ranges::size(range) <= Rows * Columns);
    if constexpr (s_kUseHeap) {
      m_data_ = new T[Rows * Columns];
    }
    std::copy_n(range.begin(), Rows * Columns, m_data_);
  }

  /**
   * @brief Creates an identity matrix.
   *
   * This method constructs an identity matrix, which is a square matrix where
   * the diagonal elements are 1, and all other elements are 0. If the matrix is
   * not square, it creates a matrix where the diagonal elements up to the
   * minimum of the number of rows and columns are set to 1.
   *
   * @return A new identity matrix.
   */
  static constexpr auto Identity() -> Matrix {
    Matrix                m(0);
    constexpr std::size_t kMin = std::min(Rows, Columns);
    for (std::size_t i = 0; i < kMin; ++i) {
      m(i, i) = 1;
    }
    return m;
  }

  /**
   * @brief Destructor for the Matrix class.
   *
   * Frees the memory allocated for the matrix.
   */
  ~Matrix() {
    if constexpr (s_kUseHeap) {
      delete[] m_data_;
    }
  }

  /**
   * @brief Provides access to a matrix element.
   *
   * @param row The row index of the element.
   * @param col The column index of the element.
   * @return A reference to the element at the specified row and column.
   */
  auto operator()(std::size_t row, std::size_t col) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_data_[row * Columns + col];
    } else {
      return m_data_[col * Rows + row];
    }
  }

  /**
   * @brief Provides constant access to a matrix element.
   *
   * @param row The row index of the element.
   * @param col The column index of the element.
   * @return A constant reference to the element at the specified row and
   * column.
   */
  auto operator()(std::size_t row, std::size_t col) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_data_[row * Columns + col];
    } else {
      return m_data_[col * Rows + row];
    }
  }

  /**
   * @brief Retrieves the element at the specified row and column as a constant
   * reference.
   *
   * This method returns a constant reference to the matrix element located at
   * the given row and column indices. It also performs an assertion to ensure
   * the indices are within bounds.
   *
   * @param row The row index of the element.
   * @param col The column index of the element.
   * @return A constant reference to the matrix element at the specified row and
   * column.
   * @note The method is marked as `[[nodiscard]]` to indicate the return value
   * should not be ignored.
   */
  [[nodiscard]] auto coeff(std::size_t row, std::size_t col) const -> const T& {
    assert(row < Rows && col < Columns && "Index out of bounds");
    return operator()(row, col);
  }

  /**
   * @brief Retrieves the element at the specified row and column as a mutable
   * reference.
   *
   * This method returns a mutable reference to the matrix element located at
   * the given row and column indices. It also performs an assertion to ensure
   * the indices are within bounds.
   *
   * @param row The row index of the element.
   * @param col The column index of the element.
   * @return A mutable reference to the matrix element at the specified row and
   * column.
   */
  auto coeffRef(std::size_t row, std::size_t col) -> T& {
    assert(row < Rows && col < Columns && "Index out of bounds");
    return operator()(row, col);
  }

  /**
   * @brief Retrieves the number of rows in the matrix.
   *
   * @return The number of rows.
   */
  static constexpr auto GetRows() -> std::size_t { return Rows; }

  /**
   * @brief Retrieves the number of columns in the matrix.
   *
   * @return The number of columns.
   */
  static constexpr auto GetColumns() -> std::size_t { return Columns; }

  /**
   * @brief Returns the total size of the matrix in bytes.
   *
   * @return The size of the matrix data in bytes.
   */
  static constexpr auto GetDataSize() -> std::size_t {
    return sizeof(T) * GetRows() * GetColumns();
  }

  /**
   * @brief Retrieves the matrix memory layout option.
   *
   * This function returns the option (either row-major or column-major) used to
   * store the matrix data.
   *
   * @return The memory layout option for the matrix (row-major or
   * column-major).
   */
  static constexpr auto GetOption() -> Options { return Option; }

  /**
   * @brief Retrieves a specific row from the matrix.
   *
   * This function returns the elements of the specified row as a vector.
   *
   * @tparam Row The index of the row to retrieve.
   * @return A vector containing the elements of the specified row.
   */
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

  /**
   * @brief Retrieves a specific column from the matrix.
   *
   * This function returns the elements of the specified column as a vector.
   *
   * @tparam Col The index of the column to retrieve.
   * @return A vector containing the elements of the specified column.
   */
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

  /**
   * @brief Sets a specific row in the matrix.
   *
   * This function updates the elements of the specified row with the values
   * from the provided vector.
   *
   * @tparam Row The index of the row to set.
   * @param vector The vector containing the new values for the row.
   */
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

  /**
   * @brief Sets a specific column in the matrix.
   *
   * This function updates the elements of the specified column with the values
   * from the provided vector.
   *
   * @tparam Col The index of the column to set.
   * @param vector The vector containing the new values for the column.
   */
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

  /**
   * @brief Sets a basis vector (row or column) in the matrix.
   *
   * This function sets a specific row or column (depending on the matrix
   * layout) with the provided vector.
   *
   * @tparam Index The index of the row/column to set.
   * @param vector The vector containing the values to set in the row/column.
   */
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

  /**
   * @brief Sets the first basis vector (X) in the matrix.
   *
   * This function updates the first basis row/column with the provided vector.
   *
   * @param vector The vector containing the values to set in the first basis
   * row/column.
   */
  void setBasisX(
      const Vector<T, (Option == Options::RowMajor ? Columns : Rows), Option>&
          vector)
    requires(Option == Options::RowMajor ? ValueLessThan<0, Columns>
                                         : ValueLessThan<0, Rows>)
  {
    setBasis<0>(vector);
  }

  /**
   * @brief Sets the second basis vector (Y) in the matrix.
   *
   * This function updates the second basis row/column with the provided vector.
   *
   * @param vector The vector containing the values to set in the second basis
   * row/column.
   */
  void setBasisY(
      const Vector<T, (Option == Options::RowMajor ? Columns : Rows), Option>&
          vector)
    requires(Option == Options::RowMajor ? ValueLessThan<1, Columns>
                                         : ValueLessThan<1, Rows>)
  {
    setBasis<1>(vector);
  }

  /**
   * @brief Sets the third basis vector (Z) in the matrix.
   *
   * This function updates the third basis row/column with the provided vector.
   *
   * @param vector The vector containing the values to set in the third basis
   * row/column.
   */
  void setBasisZ(
      const Vector<T, (Option == Options::RowMajor ? Columns : Rows), Option>&
          vector)
    requires(Option == Options::RowMajor ? ValueLessThan<2, Columns>
                                         : ValueLessThan<2, Rows>)
  {
    setBasis<2>(vector);
  }

  /**
   * @brief Retrieves the raw data pointer of the matrix.
   *
   * This function returns a pointer to the matrix data, which can be used for
   * low-level operations.
   *
   * @return A pointer to the matrix data.
   */
  auto data() -> T* { return m_data_; }

  /**
   * @brief Retrieves the raw data pointer of the matrix (const version).
   *
   * This function returns a const pointer to the matrix data for read-only
   * operations.
   *
   * @return A const pointer to the matrix data.
   */
  [[nodiscard]] auto data() const -> const T* { return m_data_; }

  /**
   * @brief Returns the transpose of the matrix.
   *
   * The transpose of a matrix is obtained by flipping it over its diagonal,
   * switching the row and column indices of the elements.
   *
   * @return A new matrix that is the transpose of the current matrix.
   */
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

  /**
   * @brief Calculates the determinant of the matrix.
   *
   * The determinant is a scalar value that can be computed from the elements of
   * a square matrix and encodes certain properties of the matrix. This function
   * uses recursive cofactor expansion for matrices larger than 2x2.
   *
   * @tparam ReturnType The type used to store the determinant value (default is
   * float).
   * @return The determinant of the matrix.
   *
   * @note This method is only defined for square matrices.
   */
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

  /**
   * @brief Calculates the inverse of the matrix using Gauss-Jordan elimination.
   *
   * This method uses Gauss-Jordan elimination to find the inverse of the
   * matrix. It augments the matrix with an identity matrix and performs row
   * operations to transform the original matrix into an identity matrix, which
   * results in the inverse matrix being formed in the augmented portion.
   *
   * @return The inverse of the matrix.
   * @note The matrix must be square.
   */
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

  /**
   * @brief Calculates the inverse of the matrix using Crout's algorithm for LU
   * decomposition.
   *
   * This method uses Crout's algorithm to perform LU decomposition of the
   * matrix and then uses forward and backward substitution to find the inverse
   * of the matrix.
   *
   * @return The inverse of the matrix.
   * @note The matrix must be square.
   * @see Matrix Inversion using LU Decomposition by Tim Bright
   *      https://www.gamedev.net/tutorials/programming/math-and-physics/matrix-inversion-using-lu-decomposition-r3637/
   *
   * @note The implementation is based on the article "Matrix Inversion using LU
   * Decomposition" by Tim Bright, published on GameDev.net on April 25, 2014.
   */
  [[nodiscard]] auto inverse() const -> Matrix {
    static_assert(Rows == Columns,
                  "Inverse is only defined for square matrices");

    Matrix<T, Rows, Columns, Option> A = *this;

    Matrix<T, Rows, Columns, Option> L(0);
    Matrix<T, Rows, Columns, Option> U(0);

    // Initialize the first column of L
    for (std::size_t i = 0; i < Rows; ++i) {
      L(i, 0) = A(i, 0);
    }

    // Initialize the scaled first row of U
    for (std::size_t j = 1; j < Columns; ++j) {
      U(0, j) = A(0, j) / L(0, 0);
    }

    // Calculate L and U elements for each subsequent column
    for (std::size_t j = 1; j < Columns - 1; ++j) {
      // Calculate L elements in column j from row j to n
      for (std::size_t i = j; i < Rows; ++i) {
        T sum = 0;
        for (std::size_t k = 0; k < j; ++k) {
          sum += L(i, k) * U(k, j);
        }
        L(i, j) = A(i, j) - sum;
      }

      // Calculate U elements in row j from column j+1 to n
      for (std::size_t k = j + 1; k < Columns; ++k) {
        T sum = 0;
        for (std::size_t i = 0; i < j; ++i) {
          sum += L(j, i) * U(i, k);
        }
        U(j, k) = (A(j, k) - sum) / L(j, j);
      }
    }

    // Calculate the final diagonal element of L
    T sum = 0;
    for (std::size_t k = 0; k < Columns - 1; ++k) {
      sum += L(Rows - 1, k) * U(k, Columns - 1);
    }
    L(Rows - 1, Columns - 1) = A(Rows - 1, Columns - 1) - sum;

    Matrix<T, Rows, Columns, Option> I
        = Matrix<T, Rows, Columns, Option>::Identity();

    // Solve LY = I for Y using forward substitution
    Matrix<T, Rows, Columns, Option> Y;
    for (std::size_t j = 0; j < Columns; ++j) {
      Y(0, j) = I(0, j) / L(0, 0);
      for (std::size_t i = 1; i < Rows; ++i) {
        T sum = 0;
        for (std::size_t k = 0; k < i; ++k) {
          sum += L(i, k) * Y(k, j);
        }
        Y(i, j) = (I(i, j) - sum) / L(i, i);
      }
    }

    // Solve UX = Y for X using backward substitution
    Matrix<T, Rows, Columns, Option> X;
    for (std::size_t j = 0; j < Columns; ++j) {
      X(Rows - 1, j) = Y(Rows - 1, j);
      for (std::int32_t i = Rows - 2; i >= 0; --i) {
        T sum = 0;
        for (std::size_t k = i + 1; k < Columns; ++k) {
          sum += U(i, k) * X(k, j);
        }
        X(i, j) = Y(i, j) - sum;
      }
    }

    return X;
  }

#endif  // LU_DECOMPOSITION_MATRIX_INVERSE

  /**
   * @brief Computes the rank of the matrix using Gaussian elimination.
   *
   * This method calculates the rank of the matrix by performing Gaussian
   * elimination on a copy of the matrix. The rank is the number of non-zero
   * rows in the row echelon form of the matrix.
   *
   * @return The rank of the matrix.
   */
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
   * @brief Calculates the Frobenius norm (magnitude) of a one-dimensional
   * matrix.
   *
   * This method computes the Frobenius norm, which is the square root of the
   * sum of the squares of all elements in the matrix. This is equivalent to the
   * Euclidean norm for vectors.
   *
   * @return The Frobenius norm of the matrix.
   */
  [[nodiscard]] auto magnitude() const -> T
    requires OneDimensional<Rows, Columns>
  {
    T                     sum              = 0;
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
   * @brief Calculates the squared Frobenius norm (squared magnitude) of a
   * one-dimensional matrix.
   *
   * This method computes the square of the Frobenius norm, which avoids the
   * square root calculation, making it faster for cases where the actual norm
   * is not required.
   *
   * @return The squared Frobenius norm of the matrix.
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
   * This method normalizes the matrix, making its Frobenius norm equal to 1 by
   * dividing all elements by the magnitude of the matrix. This method modifies
   * the matrix in-place.
   *
   * @note The matrix must not have a magnitude of zero (i.e., it must not be a
   * zero matrix).
   */
  void normalize()
    requires OneDimensional<Rows, Columns>
  {
    T mag = magnitude();
    assert(mag != 0 && "Normalization error: magnitude is zero, implying a zero matrix/vector");
    *this /= mag;
  }

  /**
   * @brief Normalizes the matrix based on its Frobenius (Euclidean) norm
   * (non-in-place).
   *
   * This method returns a new matrix that is the normalized version of the
   * current matrix. It does not modify the original matrix.
   *
   * @return A new normalized matrix.
   * @note The matrix must not have a magnitude of zero (i.e., it must not be a
   * zero matrix).
   */
  [[nodiscard]] auto normalized() const -> Matrix
    requires OneDimensional<Rows, Columns>
  {
    T mag = magnitude();
    assert(mag != 0 && "Normalization error: magnitude is zero, implying a zero matrix/vector");
    return *this / mag;
  }

  /**
   * @brief Computes the dot product of two one-dimensional matrices (vectors).
   *
   * This function calculates the dot product between two vectors (1D matrices)
   * of the same size.
   *
   * @tparam OtherRows The number of rows in the other matrix.
   * @tparam OtherColumns The number of columns in the other matrix.
   * @param other The other matrix (vector) to compute the dot product with.
   * @return The scalar result of the dot product.
   *
   * @note Both matrices must be one-dimensional and of the same size.
   */
  template <std::size_t OtherRows, std::size_t OtherColumns>
    requires OneDimensional<Rows, Columns>
          && OneDimensional<OtherRows, OtherColumns>
          && SameSize<Matrix<T, Rows, Columns>,
                      Matrix<T, OtherRows, OtherColumns>>
  [[nodiscard]] auto dot(const Matrix<T, OtherRows, OtherColumns>& other) const
      -> T {
    T                     result           = NAN;
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

  /**
   * @brief Computes the cross product of two 3D vectors.
   *
   * This function calculates the cross product between two three-dimensional
   * vectors (3x1 or 1x3 matrices).
   *
   * @tparam OtherRows The number of rows in the other matrix.
   * @tparam OtherColumns The number of columns in the other matrix.
   * @param other The other 3D vector to compute the cross product with.
   * @return A new 3D vector (matrix) representing the cross product result.
   *
   * @note Both matrices must be three-dimensional vectors.
   */
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

  /**
   * @brief Calculates the trace of the matrix.
   *
   * The trace of a matrix is the sum of its diagonal elements. This operation
   * is only defined for square matrices.
   *
   * @return The trace of the matrix (sum of diagonal elements).
   *
   * @note The matrix must be square (i.e., the number of rows must equal the
   * number of columns).
   */
  [[nodiscard]] auto trace() const -> T {
    static_assert(Rows == Columns, "Trace is only defined for square matrices");
    T sum = 0;
    for (std::size_t i = 0; i < Rows; ++i) {
      sum += this->operator()(i, i);
    }
    return sum;
  }

  /**
   * @brief Reshapes the matrix into a new matrix with different dimensions.
   *
   * This function reshapes the matrix into a new matrix with the specified
   * number of rows and columns. The total number of elements must remain the
   * same, meaning that `Rows * Columns` must equal `NewRows * NewColumns`.
   *
   * @tparam NewRows The number of rows in the reshaped matrix.
   * @tparam NewColumns The number of columns in the reshaped matrix.
   * @return A new matrix with the specified dimensions.
   *
   * @note The new matrix dimensions must have the same total size as the
   * original matrix.
   */
  template <std::size_t NewRows, std::size_t NewColumns>
  [[nodiscard]] auto reshape() const -> Matrix<T, NewRows, NewColumns, Option> {
    static_assert(
        Rows * Columns == NewRows * NewColumns,
        "New dimensions must have the same total size as the original matrix");
    Matrix<T, NewRows, NewColumns, Option> newMatrix;
    std::copy_n(m_data_, Rows * Columns, newMatrix.data());
    return newMatrix;
  }

  /**
   * @brief Adds two matrices element-wise.
   *
   * This operator performs element-wise addition of two matrices using SIMD
   * instructions if available.
   *
   * @param other The matrix to add.
   * @return A new matrix representing the result of the addition.
   */
  auto operator+(const Matrix& other) const -> Matrix {
    Matrix result  = *this;
    auto   addFunc = InstructionSet<T>::GetAddFunc();
    addFunc(result.m_data_, other.m_data_, Rows * Columns);
    return result;
  }

  /**
   * @brief Performs element-wise addition with another matrix (in-place).
   *
   * This operator modifies the current matrix by adding the corresponding
   * elements of the other matrix.
   *
   * @param other The matrix to add.
   * @return A reference to the modified matrix.
   */
  auto operator+=(const Matrix& other) -> Matrix& {
    auto addFunc = InstructionSet<T>::GetAddFunc();
    addFunc(m_data_, other.m_data_, Rows * Columns);
    return *this;
  }

  /**
   * @brief Adds a scalar to each element of the matrix.
   *
   * This operator adds a scalar value to each element of the matrix.
   *
   * @param scalar The scalar value to add.
   * @return A new matrix with the result of the addition.
   */
  auto operator+(const T& scalar) const -> Matrix {
    Matrix result        = *this;
    auto   addScalarFunc = InstructionSet<T>::GetAddScalarFunc();
    addScalarFunc(result.m_data_, scalar, Rows * Columns);
    return result;
  }

  /**
   * @brief Adds a scalar to each element of the matrix (in-place).
   *
   * This operator modifies the current matrix by adding a scalar value to each
   * element.
   *
   * @param scalar The scalar value to add.
   * @return A reference to the modified matrix.
   */
  auto operator+=(const T& scalar) -> Matrix& {
    auto addScalarFunc = InstructionSet<T>::GetAddScalarFunc();
    addScalarFunc(m_data_, scalar, Rows * Columns);
    return *this;
  }

  /**
   * @brief Subtracts another matrix element-wise.
   *
   * This operator performs element-wise subtraction of another matrix from the
   * current matrix.
   *
   * @param other The matrix to subtract.
   * @return A new matrix representing the result of the subtraction.
   */
  auto operator-(const Matrix& other) const -> Matrix {
    Matrix result  = *this;
    auto   subFunc = InstructionSet<T>::GetSubFunc();
    subFunc(result.m_data_, other.m_data_, Rows * Columns);
    return result;
  }

  /**
   * @brief Performs element-wise subtraction with another matrix (in-place).
   *
   * This operator modifies the current matrix by subtracting the corresponding
   * elements of the other matrix.
   *
   * @param other The matrix to subtract.
   * @return A reference to the modified matrix.
   */
  auto operator-=(const Matrix& other) -> Matrix& {
    auto subFunc = InstructionSet<T>::GetSubFunc();
    subFunc(m_data_, other.m_data_, Rows * Columns);
    return *this;
  }

  /**
   * @brief Subtracts a scalar from each element of the matrix.
   *
   * This operator subtracts a scalar value from each element of the matrix.
   *
   * @param scalar The scalar value to subtract.
   * @return A new matrix with the result of the subtraction.
   */
  auto operator-(const T& scalar) const -> Matrix {
    Matrix result        = *this;
    auto   subScalarFunc = InstructionSet<T>::GetSubScalarFunc();
    subScalarFunc(result.m_data_, scalar, Rows * Columns);
    return result;
  }

  /**
   * @brief Subtracts a scalar from each element of the matrix (in-place).
   *
   * This operator modifies the current matrix by subtracting a scalar value
   * from each element.
   *
   * @param scalar The scalar value to subtract.
   * @return A reference to the modified matrix.
   */
  auto operator-=(const T& scalar) -> Matrix& {
    auto subScalarFunc = InstructionSet<T>::GetSubScalarFunc();
    subScalarFunc(m_data_, scalar, Rows * Columns);
    return *this;
  }

  /**
   * @brief Negates the matrix, i.e., multiplies each element by -1.
   *
   * This operator negates the matrix by multiplying each element by -1.
   *
   * @return A new matrix representing the negated matrix.
   */
  auto operator-() const -> Matrix {
    Matrix result  = *this;
    auto   negFunc = InstructionSet<T>::GetNegFunc();
    negFunc(result.data(), Rows * Columns);
    return result;
  }

  /**
   * @brief Multiplies this matrix with another matrix and returns the result.
   *
   * This operator performs matrix multiplication, where this matrix is
   * multiplied by another matrix. The number of columns in this matrix must
   * match the number of rows in the other matrix. The result is a new matrix
   * with dimensions (Rows x ResultColumns).
   *
   * @tparam ResultColumns The number of columns in the resulting matrix.
   * @param other The matrix to multiply with this matrix.
   * @return A new matrix that is the product of the two matrices.
   */
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
   * This operator multiplies this matrix by another matrix and stores the
   * result in this matrix. This is typically used for square matrices. The
   * number of columns in this matrix must match the number of rows in the other
   * matrix.
   *
   * @tparam OtherRows The number of rows in the other matrix.
   * @tparam OtherColumns The number of columns in the other matrix.
   * @param other The matrix to multiply with this matrix.
   * @return Reference to the updated matrix after multiplication.
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

  /**
   * @brief Multiplies each element of this matrix by a scalar and returns the
   * result.
   *
   * This operator performs scalar multiplication on this matrix, where each
   * element is multiplied by the given scalar value.
   *
   * @param scalar The scalar value to multiply with.
   * @return A new matrix with each element multiplied by the scalar.
   */
  auto operator*(const T& scalar) const -> Matrix {
    Matrix result        = *this;
    auto   mulScalarFunc = InstructionSet<T>::GetMulScalarFunc();
    mulScalarFunc(result.m_data_, scalar, Rows * Columns);
    return result;
  }

  /**
   * @brief Multiplies each element of this matrix by a scalar and updates this
   * matrix with the result.
   *
   * This operator performs scalar multiplication in-place, where each element
   * of the matrix is multiplied by the given scalar value.
   *
   * @param scalar The scalar value to multiply with.
   * @return Reference to the updated matrix after scalar multiplication.
   */
  auto operator*=(const T& scalar) -> Matrix& {
    auto mulScalarFunc = InstructionSet<T>::GetMulScalarFunc();
    mulScalarFunc(m_data_, scalar, Rows * Columns);
    return *this;
  }

  /**
   * @brief Divides each element of this matrix by a scalar and returns the
   * result.
   *
   * This operator performs scalar division on this matrix, where each element
   * is divided by the given scalar value.
   *
   * @param scalar The scalar value to divide by.
   * @return A new matrix with each element divided by the scalar.
   */
  auto operator/(const T& scalar) const -> Matrix {
    Matrix result        = *this;
    auto   divScalarFunc = InstructionSet<T>::GetDivScalarFunc();
    divScalarFunc(result.m_data_, scalar, Rows * Columns);
    return result;
  }

  /**
   * @brief Divides each element of this matrix by a scalar and updates this
   * matrix with the result.
   *
   * This operator performs scalar division in-place, where each element
   * of the matrix is divided by the given scalar value.
   *
   * @param scalar The scalar value to divide by.
   * @return Reference to the updated matrix after scalar division.
   */
  auto operator/=(const T& scalar) -> Matrix& {
    auto divScalarFunc = InstructionSet<T>::GetDivScalarFunc();
    divScalarFunc(m_data_, scalar, Rows * Columns);
    return *this;
  }

  /**
   * @brief Compares this matrix with another matrix for equality.
   *
   * This operator checks whether two matrices are element-wise equal.
   * For floating-point matrices, a small epsilon value is used to account for
   * floating-point inaccuracies.
   *
   * @param other The matrix to compare with.
   * @return True if the matrices are equal, otherwise false.
   */
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

  /**
   * @brief Compares this matrix with another matrix for inequality.
   *
   * This operator checks whether two matrices are not equal.
   *
   * @param other The matrix to compare with.
   * @return True if the matrices are not equal, otherwise false.
   */
  auto operator!=(const Matrix& other) const -> bool {
    return !(*this == other);
  }

  /**
   * @brief Overloads the << operator for outputting matrix elements to an
   * output stream.
   *
   * This function outputs the matrix in a human-readable format, row by row,
   * with spaces between each element. Rows are separated by newlines.
   *
   * @param os The output stream to write to.
   * @param matrix The matrix to be output.
   * @return A reference to the output stream.
   *
   * Usage example:
   * @code
   * math::MatrixXf<2, 3, math::Options::RowMajor> m1;
   * m1 << 1, 2, 3,
   *       4, 5, 6;
   * std::cout << m1;
   * // Output:
   * // 1 2 3
   * // 4 5 6
   * @endcode
   */
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

  /**
   * @brief Helper class to initialize a matrix using the comma operator.
   *
   * This class allows initializing matrix elements in an intuitive manner using
   * the comma operator. Elements are filled row by row. If the matrix is
   * column-major, elements are filled as they are accessed in row-major order
   * for clarity.
   */
  class MatrixInitializer {
    Matrix&     m_matrix_;
    std::size_t m_row_;
    std::size_t m_col_;

    public:
    /**
     * @brief Constructs a MatrixInitializer for the given matrix.
     *
     * @param matrix The matrix to initialize.
     */
    MatrixInitializer(Matrix& matrix)
        : m_matrix_(matrix)
        , m_row_(0)
        , m_col_(0) {}

    /**
     * @brief Fills the next element of the matrix with the given value.
     *
     * This function fills the next element in the matrix. If a row is filled,
     * it moves to the next row.
     *
     * @param value The value to assign to the next element in the matrix.
     * @return A reference to the MatrixInitializer, allowing for chaining using
     * the comma operator.
     */
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
   * @brief Overloads the << operator for initializing Matrix elements using the
   * comma operator.
   *
   * This operator allows for intuitive matrix initialization by chaining
   * element assignments using the comma operator.
   *
   * Usage example for a row-major matrix:
   * @code
   * math::MatrixXf<2, 3, math::Options::RowMajor> m1;
   * m1 << 1, 2, 3,
   *       4, 5, 6;
   * // Output: 1 2 3
   * //         4 5 6
   * @endcode
   *
   * Usage example for a column-major matrix:
   * @code
   * math::MatrixXf<3, 2, math::Options::ColumnMajor> m2;
   * m2 << 1, 4,
   *       2, 5,
   *       3, 6;
   * // Output: 1 2
   * //         3 4
   * //         5 6
   * @endcode
   *
   * @param value The first value to assign to the matrix.
   * @return A MatrixInitializer to continue assigning values using the comma
   * operator.
   *
   * @note Initialization through the << operator does not necessarily follow
   * memory layout (row-major or column-major).
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
 *
 * This operator allows scalar multiplication of a matrix where the scalar value
 * is on the left-hand side. The scalar value is multiplied by each element of
 * the matrix.
 *
 * @tparam ScalarType The type of the scalar value.
 * @tparam T The data type of the matrix elements.
 * @tparam Rows The number of rows in the matrix.
 * @tparam Columns The number of columns in the matrix.
 * @tparam Option Specifies whether the matrix is stored in row-major or
 * column-major order.
 * @param scalar The scalar value to be multiplied by each element of the
 * matrix.
 * @param matrix The matrix to be multiplied by the scalar value.
 * @return A new matrix with each element multiplied by the scalar value.
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

/**
 * @brief Alias for a matrix of floats with customizable dimensions and layout.
 *
 * @tparam Rows The number of rows in the matrix.
 * @tparam Columns The number of columns in the matrix.
 * @tparam Option The memory layout (row-major or column-major).
 */
template <std::size_t Rows,
          std::size_t Columns,
          Options     Option = Options::RowMajor>
using MatrixXf = Matrix<float, Rows, Columns, Option>;

/**
 * @brief Alias for a matrix of doubles with customizable dimensions and layout.
 *
 * @tparam Rows The number of rows in the matrix.
 * @tparam Columns The number of columns in the matrix.
 * @tparam Option The memory layout (row-major or column-major).
 */
template <std::size_t Rows,
          std::size_t Columns,
          Options     Option = Options::RowMajor>
using MatrixXd = Matrix<double, Rows, Columns, Option>;

/**
 * @brief Alias for a matrix of 32-bit integers with customizable dimensions and
 * layout.
 *
 * @tparam Rows The number of rows in the matrix.
 * @tparam Columns The number of columns in the matrix.
 * @tparam Option The memory layout (row-major or column-major).
 */
template <std::size_t Rows,
          std::size_t Columns,
          Options     Option = Options::RowMajor>
using MatrixXi = Matrix<std::int32_t, Rows, Columns, Option>;

/**
 * @brief Alias for a matrix of unsigned 32-bit integers with customizable
 * dimensions and layout.
 *
 * @tparam Rows The number of rows in the matrix.
 * @tparam Columns The number of columns in the matrix.
 * @tparam Option The memory layout (row-major or column-major).
 */
template <std::size_t Rows,
          std::size_t Columns,
          Options     Option = Options::RowMajor>
using MatrixXui = Matrix<std::uint32_t, Rows, Columns, Option>;

/**
 * @brief Alias for a 2x2 matrix of floats.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix2f = MatrixXf<2, 2, Option>;

/**
 * @brief Alias for a 3x3 matrix of floats.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix3f = MatrixXf<3, 3, Option>;

/**
 * @brief Alias for a 4x4 matrix of floats.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix4f = MatrixXf<4, 4, Option>;

/**
 * @brief Alias for a 2x2 matrix of doubles.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix2d = MatrixXd<2, 2, Option>;

/**
 * @brief Alias for a 3x3 matrix of doubles.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix3d = MatrixXd<3, 3, Option>;

/**
 * @brief Alias for a 4x4 matrix of doubles.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix4d = MatrixXd<4, 4, Option>;

/**
 * @brief Alias for a 2x2 matrix of 32-bit integers.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix2i = MatrixXi<2, 2, Option>;

/**
 * @brief Alias for a 3x3 matrix of 32-bit integers.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix3i = MatrixXi<3, 3, Option>;

/**
 * @brief Alias for a 4x4 matrix of 32-bit integers.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix4i = MatrixXi<4, 4, Option>;

/**
 * @brief Alias for a 2x2 matrix of unsigned 32-bit integers.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix2ui = MatrixXui<2, 2, Option>;

/**
 * @brief Alias for a 3x3 matrix of unsigned 32-bit integers.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix3ui = MatrixXui<3, 3, Option>;

/**
 * @brief Alias for a 4x4 matrix of unsigned 32-bit integers.
 *
 * @tparam Option The memory layout (row-major or column-major).
 */
template <Options Option = Options::RowMajor>
using Matrix4ui = MatrixXui<4, 4, Option>;

}  // namespace math

#endif  // MATH_LIBRARY_MATRIX_H
