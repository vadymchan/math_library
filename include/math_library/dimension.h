/**
 * @file dimension.h
 */

#ifndef MATH_LIBRARY_DIMENSION
#define MATH_LIBRARY_DIMENSION

#include "vector.h"

namespace math {

template <typename T, unsigned int Size, Options Option = Options::RowMajor>
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

  template <typename... Args>
    requires AllSameAs<T, Args...> && ArgsSizeGreaterThanCount<1, Args...>
  Dimension(Args... args)
      : m_dataStorage_(args...) {}

  template <std::input_iterator InputIt>
  Dimension(InputIt first, InputIt last)
      : m_dataStorage_(first, last) {}

  template <std::ranges::range Range>
  Dimension(const Range& range)
      : m_dataStorage_(range) {}

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

  auto operator()(unsigned int index) const -> const T& {
    return m_dataStorage_(index);
  }

  auto operator()(unsigned int index) -> T& { return m_dataStorage_(index); }

  [[nodiscard]] auto coeff(unsigned int index) const -> const T& {
    return m_dataStorage_.coeff(index);
  }

  auto coeffRef(unsigned int index) -> T& {
    return m_dataStorage_.coeffRef(index);
  }

  static constexpr auto GetSize() -> unsigned int { return Size; }

  static constexpr auto GetOption() -> Options { return Option; }

  auto data() -> T* { return m_dataStorage_.data(); }

  [[nodiscard]] auto data() const -> const T* { return m_dataStorage_.data(); }

  private:
  Vector<T, Size, Option> m_dataStorage_;
};

// Dimension of floats
template <unsigned int Size, Options Option = Options::RowMajor>
using DimensionNf = Dimension<float, Size, Option>;

// Dimension of doubles
template <unsigned int Size, Options Option = Options::RowMajor>
using DimensionNd = Dimension<double, Size, Option>;

// Dimension of ints
template <unsigned int Size, Options Option = Options::RowMajor>
using DimensionNi = Dimension<int, Size, Option>;

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

using Dimension2Di = Dimension2D<int>;
using Dimension3Di = Dimension3D<int>;
using Dimension4Di = Dimension4D<int>;

}  // namespace math

#endif
