/**
 * @file dimension.h
 * @brief Provides the Dimension class template for representing N-dimensional
 * vectors.
 *
 * The Dimension class template allows for the representation and manipulation
 * of N-dimensional vectors, supporting various types and operations, including
 * element access, resizing, and arithmetic operations.
 */

#ifndef MATH_LIBRARY_DIMENSION_H
#define MATH_LIBRARY_DIMENSION_H

#include "vector.h"

namespace math {

/**
 * @brief Represents an N-dimensional vector.
 *
 * The Dimension class template provides functionality for working with
 * N-dimensional vectors, supporting operations like element access, arithmetic
 * operations, and comparison operators.
 *
 * @tparam T The type of the elements in the dimension (e.g., float, double,
 * int).
 * @tparam Size The number of dimensions (size of the vector).
 * @tparam Option The memory layout option (row-major or column-major). Default
 * is row-major.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
class Dimension {
  public:
  /**
   * @brief Default constructor.
   *
   * Initializes the dimension with default values.
   */
  Dimension()
      : m_dataStorage_() {}

  /**
   * @brief Constructs a Dimension with the same value for all elements.
   *
   * @param element The value to initialize all elements with.
   */
  Dimension(const T& element)
      : m_dataStorage_(element) {}

  /**
   * @brief Copy constructor.
   *
   * Creates a new Dimension by copying the data from another Dimension.
   *
   * @param other The Dimension to copy from.
   */
  Dimension(const Dimension& other)
      : m_dataStorage_(other.m_dataStorage_) {}

  /**
   * @brief Copy assignment operator.
   *
   * Copies the contents of one Dimension to another.
   *
   * @param other The Dimension to copy from.
   * @return A reference to the updated Dimension.
   */
  auto operator=(const Dimension& other) -> Dimension& {
    if (this != &other) {
      m_dataStorage_ = other.m_dataStorage_;
    }
    return *this;
  }

  /**
   * @brief Move constructor.
   *
   * Moves the contents of another Dimension to this one.
   *
   * @param other The Dimension to move from.
   */
  Dimension(Dimension&& other) noexcept
      : m_dataStorage_(std::move(other.m_dataStorage_)) {}

  /**
   * @brief Move assignment operator.
   *
   * Moves the contents of another Dimension to this one.
   *
   * @param other The Dimension to move from.
   * @return A reference to the updated Dimension.
   */
  auto operator=(Dimension&& other) noexcept -> Dimension& {
    if (this != &other) {
      m_dataStorage_ = std::move(other.m_dataStorage_);
    }
    return *this;
  }

  /**
   * @brief Constructs a Dimension from a Vector.
   *
   * @param vector The vector to initialize the dimension from.
   */
  explicit Dimension(const Vector<T, Size, Option>& vector)
      : m_dataStorage_(vector) {}

  /**
   * @brief Constructs a Dimension by moving a Vector.
   *
   * @param vector The vector to move into the dimension.
   */
  explicit Dimension(Vector<T, Size, Option>&& vector) noexcept
      : m_dataStorage_(std::move(vector)) {}

  /**
   * @brief Assigns a Vector to this Dimension.
   *
   * @param vector The vector to assign to the dimension.
   * @return A reference to the updated Dimension.
   */
  auto operator=(const Vector<T, Size, Option>& vector) -> Dimension& {
    if (m_dataStorage_ != vector) {
      m_dataStorage_ = vector;
    }
    return *this;
  }

  /**
   * @brief Constructs a Dimension from multiple arguments.
   *
   * Accepts a variable number of arguments and constructs the dimension from
   * them.
   *
   * @tparam Args Types of the arguments (must be convertible to `T`).
   * @param args The arguments to construct the dimension from.
   */
  template <typename... Args>
    requires AllConvertibleTo<T, Args...>
          && ArgsSizeGreaterThanCount<1, Args...>
  Dimension(Args... args)
      : m_dataStorage_(args...) {}

  /**
   * @brief Constructs a Dimension from an iterator range.
   *
   * Initializes the dimension with elements from the specified iterator range.
   *
   * @tparam InputIt The type of the iterator.
   * @param first The beginning of the range.
   * @param last The end of the range.
   */
  template <std::input_iterator InputIt>
  Dimension(InputIt first, InputIt last)
      : m_dataStorage_(first, last) {}

  /**
   * @brief Constructs a Dimension from a range.
   *
   * Initializes the dimension with elements from the specified range.
   *
   * @tparam Range The type of the range.
   * @param range The range to initialize the dimension from.
   */
  template <std::ranges::range Range>
  Dimension(const Range& range)
      : m_dataStorage_(range) {}

  /**
   * @brief Destructor for the Dimension class.
   */
  ~Dimension() = default;

  /**
   * @brief Returns a reference to the width (first element) of the dimension.
   *
   * @return A reference to the width.
   */
  auto width() -> T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  /**
   * @brief Returns a constant reference to the width (first element) of the
   * dimension.
   *
   * @return A constant reference to the width.
   */
  [[nodiscard]] auto width() const -> const T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  /**
   * @brief Returns a reference to the height (second element) of the dimension.
   *
   * @return A reference to the height.
   */
  auto height() -> T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  /**
   * @brief Returns a constant reference to the height (second element) of the
   * dimension.
   *
   * @return A constant reference to the height.
   */
  [[nodiscard]] auto height() const -> const T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  /**
   * @brief Returns a reference to the depth (third element) of the dimension.
   *
   * @return A reference to the depth.
   */
  auto depth() -> T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  /**
   * @brief Returns a constant reference to the depth (third element) of the
   * dimension.
   *
   * @return A constant reference to the depth.
   */
  [[nodiscard]] auto depth() const -> const T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  /**
   * @brief Accesses the element at the specified index.
   *
   * @param index The index of the element.
   * @return A reference to the element.
   */
  auto operator()(std::size_t index) const -> const T& {
    return m_dataStorage_(index);
  }

  /**
   * @brief Accesses the element at the specified index (const).
   *
   * @param index The index of the element.
   * @return A constant reference to the element.
   */
  auto operator()(std::size_t index) -> T& { return m_dataStorage_(index); }

  /**
   * @brief Returns a constant reference to the element at the specified index.
   *
   * This function provides read-only access to the element at the given index
   * in the dimension.
   *
   * @param index The index of the element to access.
   * @return A constant reference to the element at the specified index.
   */
  [[nodiscard]] auto coeff(std::size_t index) const -> const T& {
    return m_dataStorage_.coeff(index);
  }

  /**
   * @brief Returns a reference to the element at the specified index.
   *
   * This function provides writable access to the element at the given index in
   * the dimension.
   *
   * @param index The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  auto coeffRef(std::size_t index) -> T& {
    return m_dataStorage_.coeffRef(index);
  }

  /**
   * @brief Returns the size of the dimension.
   *
   * This static function returns the size (number of elements) of the
   * dimension.
   *
   * @return The size of the dimension.
   */
  static constexpr auto GetSize() -> std::size_t { return Size; }

  /**
   * @brief Returns the data size in the underlying vector.
   *
   * This static function returns the size of the data in the underlying
   * `Vector` storage.
   *
   * @return The data size of the vector.
   */
  static constexpr auto GetDataSize() -> std::size_t {
    return Vector<T, Size, Option>::GetDataSize();
  }

  /**
   * @brief Returns the memory layout option.
   *
   * This static function returns the memory layout option (either row-major or
   * column-major).
   *
   * @return The memory layout option.
   */
  static constexpr auto GetOption() -> Options { return Option; }

  /**
   * @brief Returns a resized copy of the dimension.
   *
   * This function creates and returns a new `Dimension` object with the
   * specified target size. The new dimension is a resized copy of the current
   * dimension.
   *
   * @tparam TargetSize The new size for the dimension.
   * @return A resized copy of the dimension.
   */
  template <std::size_t TargetSize>
  auto resizedCopy() const -> Dimension<T, TargetSize, Option> {
    return Dimension<T, TargetSize, Option>(
        m_dataStorage_.resizedCopy<TargetSize>());
  }

  /**
   * @brief Returns a pointer to the underlying data.
   *
   * This function returns a pointer to the underlying data stored in the
   * dimension, allowing for direct access to the elements.
   *
   * @return A pointer to the underlying data.
   */
  auto data() -> T* { return m_dataStorage_.data(); }

  /**
   * @brief Returns a constant pointer to the underlying data.
   *
   * This function returns a constant pointer to the underlying data stored in
   * the dimension, allowing for read-only access to the elements.
   *
   * @return A constant pointer to the underlying data.
   */
  [[nodiscard]] auto data() const -> const T* { return m_dataStorage_.data(); }

  /**
   * @brief Multiplies each element of the Dimension by a scalar.
   *
   * This function returns a new Dimension where each element is multiplied
   * by the given scalar value.
   *
   * @param scalar The scalar value to multiply each element by.
   * @return A new Dimension where each element is the product of the original
   * element and the scalar.
   */
  auto operator*(const T& scalar) const -> Dimension {
    return Dimension(m_dataStorage_ * scalar);
  }

  /**
   * @brief Multiplies each element of the Dimension by a scalar and updates the
   * current Dimension.
   *
   * This function multiplies each element of the current Dimension by the given
   * scalar value and updates the Dimension in-place.
   *
   * @param scalar The scalar value to multiply each element by.
   * @return A reference to the updated Dimension.
   */
  auto operator*=(const T& scalar) -> Dimension& {
    m_dataStorage_ *= scalar;
    return *this;
  }

  /**
   * @brief Divides each element of the Dimension by a scalar.
   *
   * This function returns a new Dimension where each element is divided by the
   * given scalar value.
   *
   * @param scalar The scalar value to divide each element by.
   * @return A new Dimension where each element is the quotient of the original
   * element and the scalar.
   */
  auto operator/(const T& scalar) const -> Dimension {
    return Dimension(m_dataStorage_ / scalar);
  }

  /**
   * @brief Divides each element of the Dimension by a scalar and updates the
   * current Dimension.
   *
   * This function divides each element of the current Dimension by the given
   * scalar value and updates the Dimension in-place.
   *
   * @param scalar The scalar value to divide each element by.
   * @return A reference to the updated Dimension.
   */
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

  /**
   * @brief Checks if two Dimensions are equal.
   *
   * This function compares two Dimensions to check if they are element-wise
   * equal.
   *
   * @param other The Dimension to compare against.
   * @return True if both Dimensions are equal, false otherwise.
   */
  auto operator==(const Dimension& other) const -> bool {
    return m_dataStorage_ == other.m_dataStorage_;
  }

  /**
   * @brief Checks if two Dimensions are not equal.
   *
   * This function compares two Dimensions to check if they are not element-wise
   * equal.
   *
   * @param other The Dimension to compare against.
   * @return True if the Dimensions are not equal, false otherwise.
   */
  auto operator!=(const Dimension& other) const -> bool {
    return m_dataStorage_ != other.m_dataStorage_;
  }

  /**
   * @brief Outputs the contents of the Dimension to a stream.
   *
   * This function sends the contents of the Dimension to an output stream
   * (e.g., std::cout).
   *
   * @param os The output stream.
   * @param dimension The Dimension to output.
   * @return A reference to the output stream.
   */
  friend auto operator<<(std::ostream& os, const Dimension& dimension)
      -> std::ostream& {
    return os << dimension.m_dataStorage_;
  }

  /**
   * @brief Initializes the Dimension from a single value.
   *
   * This function allows for the initialization of the Dimension with a single
   * value by using stream-like syntax (e.g., `dimension << value`).
   *
   * @param value The value to initialize the Dimension with.
   * @return A matrix initializer object for further initialization.
   */
  auto operator<<(const T& value) ->
      typename Matrix<T,
                      Option == Options::RowMajor ? 1 : Size,
                      Option == Options::RowMajor ? Size : 1,
                      Option>::MatrixInitializer {
    return m_dataStorage_ << value;
  }

  private:
  /**
   * @brief Internal storage for the elements of the Dimension.
   *
   * This Vector stores the elements of the Dimension. The size and layout of
   * the data are determined by the template parameters `Size` and `Option`,
   * where `Size` represents the number of elements (dimensions), and `Option`
   * specifies whether the data is stored in row-major or column-major order.
   */
  Vector<T, Size, Option> m_dataStorage_;
};

/**
 * @brief Performs scalar multiplication on a Dimension, where the scalar
 * value is the left-hand operand. (scalar * Dimension)
 *
 * This function multiplies a scalar by a Dimension, element-wise, and returns
 * a new Dimension with the results. The scalar is the left-hand operand in this
 * operation, and each element in the Dimension is multiplied by the scalar.
 *
 * @tparam ScalarType The type of the scalar (e.g., float, int).
 * @tparam T The type of the elements in the Dimension.
 * @tparam Size The size of the Dimension (number of elements).
 * @tparam Option The memory layout option (row-major or column-major).
 * @param scalar The scalar value to multiply with.
 * @param dimension The Dimension to be multiplied.
 * @return A new Dimension with the result of the scalar multiplication.
 */
template <typename ScalarType, typename T, std::size_t Size, Options Option>
auto operator*(const ScalarType&                 scalar,
               const Dimension<T, Size, Option>& dimension)
    -> Dimension<T, Size, Option> {
  return dimension * static_cast<T>(scalar);
}

/**
 * @brief Alias for Dimension with float elements.
 * 
 * This defines a Dimension with float elements and the specified size.
 * @tparam Size The size of the Dimension.
 * @tparam Option The memory layout option (row-major or column-major). Default is row-major.
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using DimensionNf = Dimension<float, Size, Option>;

/**
 * @brief Alias for Dimension with double elements.
 * 
 * This defines a Dimension with double elements and the specified size.
 * @tparam Size The size of the Dimension.
 * @tparam Option The memory layout option (row-major or column-major). Default is row-major.
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using DimensionNd = Dimension<double, Size, Option>;

/**
 * @brief Alias for Dimension with int (std::int32_t) elements.
 * 
 * This defines a Dimension with int elements and the specified size.
 * @tparam Size The size of the Dimension.
 * @tparam Option The memory layout option (row-major or column-major). Default is row-major.
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using DimensionNi = Dimension<std::int32_t, Size, Option>;

/**
 * @brief Alias for Dimension with unsigned int (std::uint32_t) elements.
 * 
 * This defines a Dimension with unsigned int elements and the specified size.
 * @tparam Size The size of the Dimension.
 * @tparam Option The memory layout option (row-major or column-major). Default is row-major.
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using DimensionNui = Dimension<std::uint32_t, Size, Option>;

// Templated Dimension definitions for 2D, 3D, and 4D

/**
 * @brief Alias for a 2D Dimension with the specified type.
 * 
 * This defines a 2D Dimension (size 2) for a specific type.
 * @tparam T The type of the elements in the Dimension.
 * @tparam Option The memory layout option (row-major or column-major). Default is row-major.
 */
template <typename T, Options Option = Options::RowMajor>
using Dimension2D = Dimension<T, 2, Option>;

/**
 * @brief Alias for a 3D Dimension with the specified type.
 * 
 * This defines a 3D Dimension (size 3) for a specific type.
 * @tparam T The type of the elements in the Dimension.
 * @tparam Option The memory layout option (row-major or column-major). Default is row-major.
 */
template <typename T, Options Option = Options::RowMajor>
using Dimension3D = Dimension<T, 3, Option>;

/**
 * @brief Alias for a 4D Dimension with the specified type.
 * 
 * This defines a 4D Dimension (size 4) for a specific type.
 * @tparam T The type of the elements in the Dimension.
 * @tparam Option The memory layout option (row-major or column-major). Default is row-major.
 */
template <typename T, Options Option = Options::RowMajor>
using Dimension4D = Dimension<T, 4, Option>;

// Specific data type dimensions

/**
 * @brief Alias for a 2D Dimension with float elements.
 */
using Dimension2Df = Dimension2D<float>;

/**
 * @brief Alias for a 3D Dimension with float elements.
 */
using Dimension3Df = Dimension3D<float>;

/**
 * @brief Alias for a 4D Dimension with float elements.
 */
using Dimension4Df = Dimension4D<float>;

/**
 * @brief Alias for a 2D Dimension with double elements.
 */
using Dimension2Dd = Dimension2D<double>;

/**
 * @brief Alias for a 3D Dimension with double elements.
 */
using Dimension3Dd = Dimension3D<double>;

/**
 * @brief Alias for a 4D Dimension with double elements.
 */
using Dimension4Dd = Dimension4D<double>;

/**
 * @brief Alias for a 2D Dimension with int (std::int32_t) elements.
 */
using Dimension2Di = Dimension2D<std::int32_t>;

/**
 * @brief Alias for a 3D Dimension with int (std::int32_t) elements.
 */
using Dimension3Di = Dimension3D<std::int32_t>;

/**
 * @brief Alias for a 4D Dimension with int (std::int32_t) elements.
 */
using Dimension4Di = Dimension4D<std::int32_t>;

/**
 * @brief Alias for a 2D Dimension with unsigned int (std::uint32_t) elements.
 */
using Dimension2Dui = Dimension2D<std::uint32_t>;

/**
 * @brief Alias for a 3D Dimension with unsigned int (std::uint32_t) elements.
 */
using Dimension3Dui = Dimension3D<std::uint32_t>;

/**
 * @brief Alias for a 4D Dimension with unsigned int (std::uint32_t) elements.
 */
using Dimension4Dui = Dimension4D<std::uint32_t>;

}  // namespace math

#endif  // MATH_LIBRARY_DIMENSION_H
