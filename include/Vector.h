/**
 * @file Vector.h
 */

#ifndef MATH_LIBRARY_VECTOR_H
#define MATH_LIBRARY_VECTOR_H

#include "Matrix.h"

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

  private:
  using UnderlyingType = std::conditional_t<Option == Options::RowMajor,
                                            Matrix<T, Size, 1, Option>,
                                            Matrix<T, 1, Size, Option>>;

  UnderlyingType m_dataStorage_;
};

}  // namespace math

#endif