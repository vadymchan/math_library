/**
 * @file vector.h
 */

#ifndef MATH_LIBRARY_VECTOR_H
#define MATH_LIBRARY_VECTOR_H

#include "matrix.h"

namespace math {

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
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

  template <std::size_t Rows, std::size_t Columns>
    requires OneDimensional<Rows, Columns>
  explicit Vector(const Matrix<T, Rows, Columns, Option>& matrix)
      : m_dataStorage_(matrix) {}

  template <std::size_t Rows, std::size_t Columns>
    requires OneDimensional<Rows, Columns>
  explicit Vector(Matrix<T, Rows, Columns, Option>&& matrix) noexcept
      : m_dataStorage_(std::move(matrix)) {}

  template <std::size_t Rows, std::size_t Columns>
    requires OneDimensional<Rows, Columns>
  auto operator=(const Matrix<T, Rows, Columns, Option>& matrix) -> Vector& {
    if (m_dataStorage_ != matrix) {
      m_dataStorage_ = matrix;
    }
    return *this;
  }

  template <typename... Args>
    requires AllConvertibleTo<T, Args...>
          && ArgsSizeGreaterThanCount<1, Args...>
  Vector(Args... args)
      : m_dataStorage_(static_cast<T>(args)...) {}

  template <std::input_iterator InputIt>
  Vector(InputIt first, InputIt last)
      : m_dataStorage_(first, last) {}

  template <std::ranges::range Range>
  Vector(const Range& range)
      : m_dataStorage_(range) {}

  template <typename... Elements>
  Vector(const Vector<T, Size - sizeof...(Elements), Option>& base,
         Elements&&... elements)
      : Vector() {
    static_assert((std::is_convertible_v<Elements, T> && ...),
                  "All additional elements must be convertible to the Vector's "
                  "element type");

    constexpr std::size_t kBaseSize = Size - sizeof...(Elements);

    // copy vector
    const T* baseData = base.data();
    std::copy(baseData, baseData + kBaseSize, this->data());

    // copy elements
    const T kExtra[] = {static_cast<T>(std::forward<Elements>(elements))...};
    std::copy(std::begin(kExtra), std::end(kExtra), this->data() + kBaseSize);
  }

  ~Vector() = default;

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

  auto operator()(std::size_t index) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_(0, index);
    } else {
      return m_dataStorage_(index, 0);
    }
  }

  auto operator()(std::size_t index) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_(0, index);
    } else {
      return m_dataStorage_(index, 0);
    }
  }

  [[nodiscard]] auto coeff(std::size_t index) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_.coeff(0, index);
    } else {
      return m_dataStorage_.coeff(index, 0);
    }
  }

  auto coeffRef(std::size_t index) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_.coeffRef(0, index);
    } else {
      return m_dataStorage_.coeffRef(index, 0);
    }
  }

  static constexpr auto GetSize() -> std::size_t { return Size; }

  static constexpr auto GetDataSize() -> std::size_t {
    return UnderlyingType::GetDataSize();
  }

  static constexpr auto GetOption() -> Options { return Option; }

  template <std::size_t TargetSize>
  auto resizedCopy() const -> Vector<T, TargetSize, Option> {
    Vector<T, TargetSize, Option> result;

    constexpr std::size_t numElementsToCopy
        = (TargetSize < Size) ? TargetSize : Size;
    std::copy_n(this->data(), numElementsToCopy, result.data());

    if constexpr (TargetSize > Size) {
      constexpr std::size_t numElementsToInitialize = TargetSize - Size;
      std::fill_n(result.data() + Size, numElementsToInitialize, T());
    }

    return result;
  }

  auto data() -> T* { return m_dataStorage_.data(); }

  [[nodiscard]] auto data() const -> const T* { return m_dataStorage_.data(); }

  auto magnitude() const -> T { return m_dataStorage_.magnitude(); }

  auto magnitudeSquared() const -> T {
    return m_dataStorage_.magnitudeSquared();
  }

  /**
   * @brief Normalizes the vector (in-place).
   *
   * @note This method modifies the vector itself.
   */
  void normalize() {
    T mag = magnitude();
    assert(mag != 0
           && "Normalization error: magnitude is zero, implying a zero vector");
    *this /= mag;
  }

  /**
   * @brief Normalizes the vector (non-in-place).
   *
   * @return A new normalized vector.
   */
  [[nodiscard]] auto normalized() const -> Vector {
    T mag = magnitude();
    assert(mag != 0
           && "Normalization error: magnitude is zero, implying a zero vector");
    return *this / mag;
  }

  [[nodiscard]] auto dot(const Vector& other) const -> T {
    T                     result           = NAN;
    constexpr std::size_t kVectorDimention = 1;
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
    m_dataStorage_ += other.m_dataStorage_;
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
  template <std::size_t Rows, std::size_t Columns>
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
  template <std::size_t Rows, std::size_t Columns>
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

  // TODO: for comparison consider:
  // - comparison for vector with different dimension
  // - optional tolerance parameter (useful for floating point types)
  // - comparison for different major (row-major and column-major comparison)

  /**
   * @brief Lexicographical less than operator.
   *
   * Compares two vectors lexicographically.
   *
   * @param other The vector to compare against.
   * @return True if this vector is lexicographically less than the other
   * vector, false otherwise.
   */
  auto operator<(const Vector& other) const -> bool {
    auto cmpFunc = InstructionSet<T>::GetCmpFunc();
    return cmpFunc(this->data(), other.data(), Size) == -1;
  }

  /**
   * @brief Lexicographical greater than operator.
   *
   * Compares two vectors lexicographically.
   *
   * @param other The vector to compare against.
   * @return True if this vector is lexicographically greater than the other
   * vector, false otherwise.
   */
  auto operator>(const Vector& other) const -> bool {
    auto cmpFunc = InstructionSet<T>::GetCmpFunc();
    return cmpFunc(this->data(), other.data(), Size) == 1;
  }

  /**
   * @brief Lexicographical less than or equal to operator.
   *
   * Compares two vectors lexicographically.
   *
   * @param other The vector to compare against.
   * @return True if this vector is lexicographically less than or equal to the
   * other vector, false otherwise.
   */
  auto operator<=(const Vector& other) const -> bool {
    auto cmpFunc = InstructionSet<T>::GetCmpFunc();
    auto result  = cmpFunc(this->data(), other.data(), Size);
    return result == -1 || result == 0;
  }

  /**
   * @brief Lexicographical greater than or equal to operator.
   *
   * Compares two vectors lexicographically.
   *
   * @param other The vector to compare against.
   * @return True if this vector is lexicographically greater than or equal to
   * the other vector, false otherwise.
   */
  auto operator>=(const Vector& other) const -> bool {
    auto cmpFunc = InstructionSet<T>::GetCmpFunc();
    auto result  = cmpFunc(this->data(), other.data(), Size);
    return result == 1 || result == 0;
  }

  auto operator==(const Vector& other) const -> bool {
    return m_dataStorage_ == other.m_dataStorage_;
  }

  auto operator!=(const Vector& other) const -> bool {
    return m_dataStorage_ != other.m_dataStorage_;
  }

  friend auto operator<<(std::ostream& os, const Vector& vector)
      -> std::ostream& {
    for (std::size_t i = 0; i < Size; ++i) {
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
    Vector&     m_vector_;
    std::size_t m_index_;

    public:
    VectorInitializer(Vector& vector, std::size_t index)
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
            std::size_t Rows1,
            std::size_t Columns1,
            Options     Option1,
            std::size_t Size1>
    requires ValueEqualTo<Columns1, Size1> && (Option1 == Options::ColumnMajor)
  friend auto operator*(const Matrix<T1, Rows1, Columns1, Option1>& matrix,
                        const Vector<T1, Size1, Option1>&           vector)
      -> Vector<T1, Rows1, Option1>;

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
          std::size_t Rows,
          std::size_t Columns,
          Options     Option,
          std::size_t Size>
  requires ValueEqualTo<Columns, Size> && (Option == Options::ColumnMajor)
auto operator*(const Matrix<T, Rows, Columns, Option>& matrix,
               const Vector<T, Size, Option>&          vector)
    -> Vector<T, Rows, Option> {
  return Vector<T, Rows, Option>(matrix * vector.m_dataStorage_);
}

/**
 * @brief Performs scalar multiplication on a Vector, where the scalar value is
 * the left-hand operand. (scalar * Vector)
 */
template <typename ScalarType, typename T, std::size_t Size, Options Option>
auto operator*(const ScalarType& scalar, const Vector<T, Size, Option>& vector)
    -> Vector<T, Size, Option> {
  return vector * static_cast<T>(scalar);
}

// Vector of floats
template <std::size_t Size, Options Option = Options::RowMajor>
using VectorNf = Vector<float, Size, Option>;

// Vector of doubles
template <std::size_t Size, Options Option = Options::RowMajor>
using VectorNd = Vector<double, Size, Option>;

// Vector of ints
template <std::size_t Size, Options Option = Options::RowMajor>
using VectorNi = Vector<std::int32_t, Size, Option>;

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

using Vector2Di = Vector2D<std::int32_t>;
using Vector3Di = Vector3D<std::int32_t>;
using Vector4Di = Vector4D<std::int32_t>;

}  // namespace math

#endif