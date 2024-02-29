/**
 * @file vector.h
 */

#ifndef MATH_LIBRARY_VECTOR_H
#define MATH_LIBRARY_VECTOR_H

#include "matrix.h"

namespace math {

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
class Vector {
  public:
  Vector()
      : m_dataStorage_() {}

  Vector(const T& element)
      : m_dataStorage_(element) {}

  Vector(const Vector& other)
      : m_dataStorage_(other.m_dataStorage_) {}

  auto operator=(const Vector& other) -> Vector& {
    if (this != &other) {
      m_dataStorage_ = other.m_dataStorage_;
    }
    return *this;
  }

  Vector(Vector&& other) noexcept
      : m_dataStorage_(std::move(other.m_dataStorage_)) {}

  auto operator=(Vector&& other) noexcept -> Vector& {
    if (this != &other) {
      m_dataStorage_ = std::move(other.m_dataStorage_);
    }
    return *this;
  }

  template <unsigned int Rows, unsigned int Columns>
    requires OneDimensional<Rows, Columns>
  explicit Vector(const Matrix<T, Rows, Columns, Option>& matrix)
      : m_dataStorage_(matrix) {}

  template <unsigned int Rows, unsigned int Columns>
    requires OneDimensional<Rows, Columns>
  explicit Vector(Matrix<T, Rows, Columns, Option>&& matrix) noexcept
      : m_dataStorage_(std::move(matrix)) {}

  template <unsigned int Rows, unsigned int Columns>
    requires OneDimensional<Rows, Columns>
  auto operator=(const Matrix<T, Rows, Columns, Option>& matrix) -> Vector& {
    if (m_dataStorage_ != matrix) {
      m_dataStorage_ = matrix;
    }
    return *this;
  }

  template <typename... Args>
    requires AllSameAs<T, Args...> && ArgsSizeGreaterThanCount<1, Args...>
  Vector(Args... args)
      : m_dataStorage_(args...) {}

  template <std::input_iterator InputIt>
  Vector(InputIt first, InputIt last)
      : m_dataStorage_(first, last) {}

  template <std::ranges::range Range>
  Vector(const Range& range)
      : m_dataStorage_(range) {}

  auto x() -> T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  [[nodiscard]] auto x() const -> const T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  auto y() -> T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  [[nodiscard]] auto y() const -> const T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  auto z() -> T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  [[nodiscard]] auto z() const -> const T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  auto w() -> T&
    requires ValueAtLeast<Size, 4>
  {
    return operator()(3);
  }

  [[nodiscard]] auto w() const -> const T&
    requires ValueAtLeast<Size, 4>
  {
    return operator()(3);
  }

  auto operator()(unsigned int index) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_(0, index);
    } else {
      return m_dataStorage_(index, 0);
    }
  }

  auto operator()(unsigned int index) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_(0, index);
    } else {
      return m_dataStorage_(index, 0);
    }
  }

  [[nodiscard]] auto coeff(unsigned int index) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_.coeff(0, index);
    } else {
      return m_dataStorage_.coeff(index, 0);
    }
  }

  auto coeffRef(unsigned int index) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_.coeffRef(0, index);
    } else {
      return m_dataStorage_.coeffRef(index, 0);
    }
  }

  static constexpr auto GetSize() -> unsigned int { return Size; }

  static constexpr auto GetOption() -> Options { return Option; }

  auto data() -> T* { return m_dataStorage_.data(); }

  [[nodiscard]] auto data() const -> const T* { return m_dataStorage_.data(); }

  auto magnitude() -> T { return m_dataStorage_.magnitude(); }

#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE

  void normalize() { m_dataStorage_.normalize(); }

#else

  [[nodiscard]] auto normalize() const -> Vector {
    return Vector(m_dataStorage_.normalize());
  }

#endif

  [[nodiscard]] auto dot(const Vector& other) const -> T {
    float                  result           = NAN;
    constexpr unsigned int kVectorDimention = 1;
    auto mulFunc = InstructionSet<T>::template GetMulFunc<Option>();
    mulFunc(&result,
            this->data(),
            other.data(),
            kVectorDimention,
            kVectorDimention,
            Size);
    return result;
  }

  [[nodiscard]] auto cross(const Vector& other) const -> Vector
    requires ValueEqualTo<Size, 3>
  {
    Vector result;

    result.x() = this->y() * other.z() - this->z() * other.y();
    result.y() = this->z() * other.x() - this->x() * other.z();
    result.z() = this->x() * other.y() - this->y() * other.x();

    return result;
  }

  auto operator+(const Vector& other) const -> Vector {
    return Vector(m_dataStorage_ + other.m_dataStorage_);
  }

  auto operator+=(const Vector& other) -> Vector& {
    m_dataStorage_ += other;
    return *this;
  }

  auto operator+(const T& scalar) const -> Vector {
    return Vector(m_dataStorage_ + scalar);
  }

  auto operator+=(const T& scalar) -> Vector& {
    m_dataStorage_ += scalar;
    return *this;
  }

  auto operator-(const Vector& other) const -> Vector {
    return Vector(m_dataStorage_ - other.m_dataStorage_);
  }

  auto operator-=(const Vector& other) -> Vector& {
    m_dataStorage_ -= other;
    return *this;
  }

  auto operator-(const T& scalar) const -> Vector {
    return Vector(m_dataStorage_ - scalar);
  }

  auto operator-() const -> Vector { return Vector(-m_dataStorage_); }

  auto operator-=(const T& scalar) -> Vector& {
    m_dataStorage_ -= scalar;
    return *this;
  }

  /**
   * @brief Multiplies the vector by a matrix in a row-major context (vector *
   * matrix).
   *
   * Usage example:
   *    Vector<float, 2> vec(1.0f, 2.0f);
   *    Matrix<float, 2, 3> mat = { '1, 2, 3,' '4, 5, 6' };
   *    auto result = vec * mat; // result is a Vector<float, 3> { 9, 12, 15 }
   *
   * @note This function is only for row-major vectors where the
   *       size of the vector equals the number of rows in the matrix.
   */
  template <unsigned int Rows, unsigned int Columns>
    requires ValueEqualTo<Rows, Size> && (Option == Options::RowMajor)
  auto operator*(const Matrix<T, Rows, Columns, Option>& matrix) const
      -> Vector<T, Columns, Option> {
    return Vector<T, Columns, Option>(m_dataStorage_ * matrix);
  }

  // clang-format off

  /**
   * @brief Matrix multiplication-assignment operator
   *
   * This operator multiplies the current matrix with the given one.
   * Note: This function only works when the matrices have the same dimensions
   * and squared.
   *
   */
  template <unsigned int Rows, unsigned int Columns>
    requires SquaredMatrix<Matrix<T, Rows, Columns, Option>>
          && ((ValueEqualTo<Rows, Size> && Option == Options::RowMajor)
              || (ValueEqualTo<Columns, Size> && Option == Options::ColumnMajor))
  auto operator*=(const Matrix<T, Rows, Columns, Option>& matrix) -> Vector& {
    m_dataStorage_ *= matrix;
    return *this;
  }

  // clang-format on

  auto operator*(const T& scalar) const -> Vector {
    return Vector(m_dataStorage_ * scalar);
  }

  auto operator*=(const T& scalar) -> Vector& {
    m_dataStorage_ *= scalar;
    return *this;
  }

  auto operator/(const T& scalar) const -> Vector {
    return Vector(m_dataStorage_ / scalar);
  }

  auto operator/=(const T& scalar) -> Vector& {
    m_dataStorage_ /= scalar;
    return *this;
  }

  auto operator==(const Vector& other) const -> bool {
    return m_dataStorage_ == other.m_dataStorage_;
  }

  auto operator!=(const Vector& other) const -> bool {
    return m_dataStorage_ != other.m_dataStorage_;
  }

  friend auto operator<<(std::ostream& os, const Vector& vector)
      -> std::ostream& {
    for (int i = 0; i < Size; ++i) {
      os << vector(i) << ' ';
    }
    os << '\n';
    return os;
  }

#ifdef FEATURE_VECTOR_INITIALIZER

  /**
   * @brief Helper class for initializing Vector elements using the comma
   * operator (operator,).
   *
   * Usage example:
   *    Vector<float, 3> vec;
   *    vec << 1.0f, 2.0f, 3.0f; // Initializes vec to [1.0, 2.0, 3.0]
   */
  class VectorInitializer {
    Vector&      m_vector_;
    unsigned int m_index_;

    public:
    VectorInitializer(Vector& vector, unsigned int index)
        : m_vector_(vector)
        , m_index_(index) {}

    auto operator,(const T& value) -> VectorInitializer& {
      if (m_index_ < Size) {
        m_vector_(m_index_++) = value;
      }
      return *this;
    }
  };

  auto operator<<(const T& value) -> VectorInitializer {
    this->operator()(0) = value;
    return VectorInitializer(*this, 1);
  }

#else

  auto operator<<(const T& value) ->
      typename Matrix<T,
                      Option == Options::RowMajor ? 1 : Size,
                      Option == Options::RowMajor ? Size : 1,
                      Option>::MatrixInitializer {
    return m_dataStorage_ << value;
  }

#endif  // FEATURE_VECTOR_INITIALIZER

  template <typename T1,
            unsigned int Rows,
            unsigned int Columns,
            Options      Option1,
            unsigned int Size1>
  friend auto operator*(const Matrix<T1, Rows, Columns, Option1>& matrix,
                        const Vector<T1, Size1, Option1>&         vector)
      -> Vector<T1, Rows, Option1>;

  private:
  using UnderlyingType = std::conditional_t<Option == Options::RowMajor,
                                            Matrix<T, 1, Size, Option>,
                                            Matrix<T, Size, 1, Option>>;

  UnderlyingType m_dataStorage_;
};

/**
 * @brief Multiplies a matrix by a vector in a column-major context (matrix *
 * vector).
 *
 * Usage example:
 *    Matrix<float, 3, 2> mat = { '1, 2', '3, 4', '5, 6' };
 *    Vector<float, 2> vec(1.0f, 2.0f);
 *    auto result = mat * vec; // result is a Vector<float, 3> { 5, 11, 17 }
 *
 * @note This function is only for column-major matrices where the number of
 * columns in the matrix equals the size of the vector.
 */
template <typename T,
          unsigned int Rows,
          unsigned int Columns,
          Options      Option,
          unsigned int Size>
  requires ValueEqualTo<Columns, Size> && (Option == Options::ColumnMajor)
inline auto operator*(const Matrix<T, Rows, Columns, Option>& matrix,
                      const Vector<T, Size, Option>&          vector)
    -> Vector<T, Rows, Option> {
  return Vector<T, Rows, Option>(matrix * vector.m_dataStorage_);
}

// Vector of floats
template <unsigned int Size, Options Option = Options::RowMajor>
using VectorNf = Vector<float, Size, Option>;

// Vector of doubles
template <unsigned int Size, Options Option = Options::RowMajor>
using VectorNd = Vector<double, Size, Option>;

// Vector of ints
template <unsigned int Size, Options Option = Options::RowMajor>
using VectorNi = Vector<int, Size, Option>;

// Templated Vector 2D
template <typename T, Options Option = Options::RowMajor>
using Vector2D = Vector<T, 2, Option>;

// Templated Vector 3D
template <typename T, Options Option = Options::RowMajor>
using Vector3D = Vector<T, 3, Option>;

// Templated Vector 4D
template <typename T, Options Option = Options::RowMajor>
using Vector4D = Vector<T, 4, Option>;

// Specific data type vectors
using Vector2Df = Vector2D<float>;
using Vector3Df = Vector3D<float>;
using Vector4Df = Vector4D<float>;

using Vector2Dd = Vector2D<double>;
using Vector3Dd = Vector3D<double>;
using Vector4Dd = Vector4D<double>;

using Vector2Di = Vector2D<int>;
using Vector3Di = Vector3D<int>;
using Vector4Di = Vector4D<int>;

}  // namespace math

#endif