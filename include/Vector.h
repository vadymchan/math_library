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

  auto x() -> T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  auto x() const -> const T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  auto y() -> T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  auto y() const -> const T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  auto z() -> T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  auto z() const -> const T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  auto w() -> T&
    requires ValueAtLeast<Size, 4>
  {
    return operator()(3);
  }

  auto w() const -> const T&
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

  // TODO: add condition conmilation to normalize
  [[nodiscard]] auto normalize() const -> Vector {
    return Vector(m_dataStorage_.normalize());
  }

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

  auto operator-=(const T& scalar) -> Vector& {
    m_dataStorage_ -= scalar;
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
  private:
  using UnderlyingType = std::conditional_t<Option == Options::RowMajor,
                                            Matrix<T, 1, Size, Option>,
                                            Matrix<T, Size, 1, Option>>;

  UnderlyingType m_dataStorage_;
};

}  // namespace math

#endif