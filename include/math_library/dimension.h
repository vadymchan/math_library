/**
 * @file dimension.h
 */

#ifndef MATH_LIBRARY_DIMENSION_H
#define MATH_LIBRARY_DIMENSION_H

#include "vector.h"

namespace math {

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
class Dimension {
  public:
  Dimension()
      : m_dataStorage_() {}

  Dimension(const T& element)
      : m_dataStorage_(element) {}

  Dimension(const Dimension& other)
      : m_dataStorage_(other.m_dataStorage_) {}

  auto operator=(const Dimension& other) -> Dimension& {
    if (this != &other) {
      m_dataStorage_ = other.m_dataStorage_;
    }
    return *this;
  }

  Dimension(Dimension&& other) noexcept
      : m_dataStorage_(std::move(other.m_dataStorage_)) {}

  auto operator=(Dimension&& other) noexcept -> Dimension& {
    if (this != &other) {
      m_dataStorage_ = std::move(other.m_dataStorage_);
    }
    return *this;
  }

  explicit Dimension(const Vector<T, Size, Option>& vector)
      : m_dataStorage_(vector) {}

  explicit Dimension(Vector<T, Size, Option>&& vector) noexcept
      : m_dataStorage_(std::move(vector)) {}

  auto operator=(const Vector<T, Size, Option>& vector) -> Dimension& {
    if (m_dataStorage_ != vector) {
      m_dataStorage_ = vector;
    }
    return *this;
  }

  template <typename... Args>
    requires AllConvertibleTo<T, Args...>
          && ArgsSizeGreaterThanCount<1, Args...>
  Dimension(Args... args)
      : m_dataStorage_(args...) {}

  template <std::input_iterator InputIt>
  Dimension(InputIt first, InputIt last)
      : m_dataStorage_(first, last) {}

  template <std::ranges::range Range>
  Dimension(const Range& range)
      : m_dataStorage_(range) {}

  ~Dimension() = default;

  auto width() -> T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  [[nodiscard]] auto width() const -> const T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  auto height() -> T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  [[nodiscard]] auto height() const -> const T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  auto depth() -> T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  [[nodiscard]] auto depth() const -> const T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  auto operator()(std::size_t index) const -> const T& {
    return m_dataStorage_(index);
  }

  auto operator()(std::size_t index) -> T& { return m_dataStorage_(index); }

  [[nodiscard]] auto coeff(std::size_t index) const -> const T& {
    return m_dataStorage_.coeff(index);
  }

  auto coeffRef(std::size_t index) -> T& {
    return m_dataStorage_.coeffRef(index);
  }

  static constexpr auto GetSize() -> std::size_t { return Size; }

  static constexpr auto GetDataSize() -> std::size_t {
    return Vector<T, Size, Option>::GetDataSize();
  }

  static constexpr auto GetOption() -> Options { return Option; }

  template <std::size_t TargetSize>
  auto resizedCopy() const -> Dimension<T, TargetSize, Option> {
    return Dimension<T, TargetSize, Option>(
        m_dataStorage_.resizedCopy<TargetSize>());
  }

  auto data() -> T* { return m_dataStorage_.data(); }

  [[nodiscard]] auto data() const -> const T* { return m_dataStorage_.data(); }

  auto operator*(const T& scalar) const -> Dimension {
    return Dimension(m_dataStorage_ * scalar);
  }

  auto operator*=(const T& scalar) -> Dimension& {
    m_dataStorage_ *= scalar;
    return *this;
  }

  auto operator/(const T& scalar) const -> Dimension {
    return Dimension(m_dataStorage_ / scalar);
  }

  auto operator/=(const T& scalar) -> Dimension& {
    m_dataStorage_ /= scalar;
    return *this;
  }

  /**
   * @brief Lexicographical less than operator.
   *
   * Compares two dimensions lexicographically.
   *
   * @param other The dimension to compare against.
   * @return True if this dimension is lexicographically less than the other
   * dimension, false otherwise.
   */
  auto operator<(const Dimension& other) const -> bool {
    return this->m_dataStorage_ < other.m_dataStorage_;
  }

  /**
   * @brief Lexicographical greater than operator.
   *
   * Compares two dimensions lexicographically.
   *
   * @param other The dimension to compare against.
   * @return True if this dimension is lexicographically greater than the other
   * dimension, false otherwise.
   */
  auto operator>(const Dimension& other) const -> bool {
    return this->m_dataStorage_ > other.m_dataStorage_;
  }

  /**
   * @brief Lexicographical less than or equal to operator.
   *
   * Compares two dimensions lexicographically.
   *
   * @param other The dimension to compare against.
   * @return True if this dimension is lexicographically less than or equal to
   * the other dimension, false otherwise.
   */
  auto operator<=(const Dimension& other) const -> bool {
    return this->m_dataStorage_ <= other.m_dataStorage_;
  }

  /**
   * @brief Lexicographical greater than or equal to operator.
   *
   * Compares two dimensions lexicographically.
   *
   * @param other The dimension to compare against.
   * @return True if this dimension is lexicographically greater than or equal
   * to the other dimension, false otherwise.
   */
  auto operator>=(const Dimension& other) const -> bool {
    return this->m_dataStorage_ >= other.m_dataStorage_;
  }

  auto operator==(const Dimension& other) const -> bool {
    return m_dataStorage_ == other.m_dataStorage_;
  }

  auto operator!=(const Dimension& other) const -> bool {
    return m_dataStorage_ != other.m_dataStorage_;
  }

  friend auto operator<<(std::ostream& os, const Dimension& dimension)
      -> std::ostream& {
    return os << dimension.m_dataStorage_;
  }

  auto operator<<(const T& value) ->
      typename Matrix<T,
                      Option == Options::RowMajor ? 1 : Size,
                      Option == Options::RowMajor ? Size : 1,
                      Option>::MatrixInitializer {
    return m_dataStorage_ << value;
  }

  private:
  Vector<T, Size, Option> m_dataStorage_;
};

/**
 * @brief Performs scalar multiplication on a Dimension, where the scalar
 value is
 * the left-hand operand. (scalar * Dimension)
 */
template <typename ScalarType, typename T, std::size_t Size, Options Option>
auto operator*(const ScalarType&                 scalar,
               const Dimension<T, Size, Option>& dimension)
    -> Dimension<T, Size, Option> {
  return dimension * static_cast<T>(scalar);
}

// Dimension of floats
template <std::size_t Size, Options Option = Options::RowMajor>
using DimensionNf = Dimension<float, Size, Option>;

// Dimension of doubles
template <std::size_t Size, Options Option = Options::RowMajor>
using DimensionNd = Dimension<double, Size, Option>;

// Dimension of ints
template <std::size_t Size, Options Option = Options::RowMajor>
using DimensionNi = Dimension<std::int32_t, Size, Option>;

// Templated Dimension 2D
template <typename T, Options Option = Options::RowMajor>
using Dimension2D = Dimension<T, 2, Option>;

// Templated Dimension 3D
template <typename T, Options Option = Options::RowMajor>
using Dimension3D = Dimension<T, 3, Option>;

// Templated Dimension 4D
template <typename T, Options Option = Options::RowMajor>
using Dimension4D = Dimension<T, 4, Option>;

// Specific data type dimension
using Dimension2Df = Dimension2D<float>;
using Dimension3Df = Dimension3D<float>;
using Dimension4Df = Dimension4D<float>;

using Dimension2Dd = Dimension2D<double>;
using Dimension3Dd = Dimension3D<double>;
using Dimension4Dd = Dimension4D<double>;

using Dimension2Di = Dimension2D<std::int32_t>;
using Dimension3Di = Dimension3D<std::int32_t>;
using Dimension4Di = Dimension4D<std::int32_t>;

}  // namespace math

#endif
