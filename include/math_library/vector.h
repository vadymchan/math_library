/**
 * @file vector.h
 * @brief Defines a templated Vector class for mathematical operations.
 *
 * The `Vector` class is a flexible container for storing and manipulating
 * elements in a one-dimensional array (vector). It provides methods for
 * basic vector arithmetic, dot and cross products, and scalar operations.
 */

#ifndef MATH_LIBRARY_VECTOR_H
#define MATH_LIBRARY_VECTOR_H

#include "matrix.h"

namespace math {

/**
 * @brief A templated class for representing vectors.
 *
 * This class supports common vector operations like addition, subtraction,
 * scalar multiplication, and division. It also supports vector-specific
 * operations like dot product and cross product. The vector can be
 * initialized with a range, list of elements, or by copying another vector or
 * matrix.
 *
 * @tparam T The type of elements stored in the vector.
 * @tparam Size The number of elements in the vector.
 * @tparam Option The layout option (either RowMajor or ColumnMajor).
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
class Vector {
  public:
  /**
   * @brief Default constructor that initializes a vector with default values.
   */
  Vector()
      : m_dataStorage_() {}

  /**
   * @brief Constructs a vector where all elements are initialized with a single
   * value.
   *
   * @param element The value to initialize all elements of the vector.
   */
  Vector(const T& element)
      : m_dataStorage_(element) {}

  /**
   * @brief Copy constructor for creating a vector from another vector.
   *
   * @param other The vector to copy from.
   */
  Vector(const Vector& other)
      : m_dataStorage_(other.m_dataStorage_) {}

  /**
   * @brief Copy assignment operator for assigning one vector to another.
   *
   * @param other The vector to copy from.
   * @return A reference to the current vector.
   */
  auto operator=(const Vector& other) -> Vector& {
    if (this != &other) {
      m_dataStorage_ = other.m_dataStorage_;
    }
    return *this;
  }

  /**
   * @brief Move constructor for transferring ownership of another vector's
   * data.
   *
   * @param other The vector to move from.
   */
  Vector(Vector&& other) noexcept
      : m_dataStorage_(std::move(other.m_dataStorage_)) {}

  /**
   * @brief Move assignment operator for transferring ownership of another
   * vector's data.
   *
   * @param other The vector to move from.
   * @return A reference to the current vector.
   */
  auto operator=(Vector&& other) noexcept -> Vector& {
    if (this != &other) {
      m_dataStorage_ = std::move(other.m_dataStorage_);
    }
    return *this;
  }

  /**
   * @brief Constructs a vector from a one-dimensional matrix.
   *
   * @tparam Rows The number of rows in the matrix.
   * @tparam Columns The number of columns in the matrix.
   * @param matrix The one-dimensional matrix to initialize the vector.
   */
  template <std::size_t Rows, std::size_t Columns>
    requires OneDimensional<Rows, Columns>
  explicit Vector(const Matrix<T, Rows, Columns, Option>& matrix)
      : m_dataStorage_(matrix) {}

  /**
   * @brief Constructs a vector by moving data from a one-dimensional matrix.
   *
   * This constructor initializes a vector by moving data from a matrix that is
   * constrained to be one-dimensional (either a row or a column).
   *
   * @tparam Rows The number of rows in the matrix.
   * @tparam Columns The number of columns in the matrix.
   * @param matrix The matrix to move data from.
   */
  template <std::size_t Rows, std::size_t Columns>
    requires OneDimensional<Rows, Columns>
  explicit Vector(Matrix<T, Rows, Columns, Option>&& matrix) noexcept
      : m_dataStorage_(std::move(matrix)) {}

  /**
   * @brief Assigns a one-dimensional matrix to the vector.
   *
   * This assignment operator allows for assigning a matrix to the vector, where
   * the matrix is constrained to be one-dimensional (either a row or a column).
   *
   * @tparam Rows The number of rows in the matrix.
   * @tparam Columns The number of columns in the matrix.
   * @param matrix The matrix to assign to the vector.
   * @return A reference to the current vector.
   */
  template <std::size_t Rows, std::size_t Columns>
    requires OneDimensional<Rows, Columns>
  auto operator=(const Matrix<T, Rows, Columns, Option>& matrix) -> Vector& {
    if (m_dataStorage_ != matrix) {
      m_dataStorage_ = matrix;
    }
    return *this;
  }

  /**
   * @brief Constructs a vector from a list of arguments.
   *
   * This constructor allows the initialization of a vector using a variadic
   * list of arguments, where each argument is converted to the vector's element
   * type.
   *
   * @tparam Args The types of the arguments.
   * @param args The values to initialize the vector with.
   */
  template <typename... Args>
    requires AllConvertibleTo<T, Args...>
          && ArgsSizeGreaterThanCount<1, Args...>
  Vector(Args... args)
      : m_dataStorage_(static_cast<T>(args)...) {}

  /**
   * @brief Constructs a vector from an input iterator range.
   *
   * This constructor initializes the vector with a range defined by two input
   * iterators.
   *
   * @tparam InputIt The type of the input iterators.
   * @param first The iterator pointing to the beginning of the range.
   * @param last The iterator pointing to the end of the range.
   */
  template <std::input_iterator InputIt>
  Vector(InputIt first, InputIt last)
      : m_dataStorage_(first, last) {}

  /**
   * @brief Constructs a vector from a range object.
   *
   * This constructor allows the initialization of a vector from any object that
   * satisfies the `std::ranges::range` concept, such as standard containers.
   *
   * @tparam Range The type of the range object.
   * @param range The range object to initialize the vector with.
   */
  template <std::ranges::range Range>
  Vector(const Range& range)
      : m_dataStorage_(range) {}

  /**
   * @brief Constructs a vector by appending additional elements to an existing
   * vector.
   *
   * This constructor allows for creating a vector by copying an existing vector
   * of smaller size and appending additional elements to it.
   *
   * @tparam Elements The types of the additional elements to append.
   * @param base The base vector to copy from.
   * @param elements The additional elements to append to the vector.
   */
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

  /**
   * @brief Destructor for the Vector class.
   */
  ~Vector() = default;

  /**
   * @brief Returns a reference to the x component of the vector.
   *
   * This method provides access to the first element of the vector.
   * It is available only if the vector has at least one element.
   *
   * @return A reference to the x component (first element).
   */
  auto x() -> T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  /**
   * @brief Returns a constant reference to the x component of the vector.
   *
   * This method provides read-only access to the first element of the vector.
   * It is available only if the vector has at least one element.
   *
   * @return A constant reference to the x component (first element).
   */
  [[nodiscard]] auto x() const -> const T&
    requires ValueAtLeast<Size, 1>
  {
    return operator()(0);
  }

  /**
   * @brief Returns a reference to the y component of the vector.
   *
   * This method provides access to the second element of the vector.
   * It is available only if the vector has at least two elements.
   *
   * @return A reference to the y component (second element).
   */
  auto y() -> T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  /**
   * @brief Returns a constant reference to the y component of the vector.
   *
   * This method provides read-only access to the second element of the vector.
   * It is available only if the vector has at least two elements.
   *
   * @return A constant reference to the y component (second element).
   */
  [[nodiscard]] auto y() const -> const T&
    requires ValueAtLeast<Size, 2>
  {
    return operator()(1);
  }

  /**
   * @brief Returns a reference to the z component of the vector.
   *
   * This method provides access to the third element of the vector.
   * It is available only if the vector has at least three elements.
   *
   * @return A reference to the z component (third element).
   */
  auto z() -> T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  /**
   * @brief Returns a constant reference to the z component of the vector.
   *
   * This method provides read-only access to the third element of the vector.
   * It is available only if the vector has at least three elements.
   *
   * @return A constant reference to the z component (third element).
   */
  [[nodiscard]] auto z() const -> const T&
    requires ValueAtLeast<Size, 3>
  {
    return operator()(2);
  }

  /**
   * @brief Returns a reference to the w component of the vector.
   *
   * This method provides access to the fourth element of the vector.
   * It is available only if the vector has at least four elements.
   *
   * @return A reference to the w component (fourth element).
   */
  auto w() -> T&
    requires ValueAtLeast<Size, 4>
  {
    return operator()(3);
  }

  /**
   * @brief Returns a constant reference to the w component of the vector.
   *
   * This method provides read-only access to the fourth element of the vector.
   * It is available only if the vector has at least four elements.
   *
   * @return A constant reference to the w component (fourth element).
   */
  [[nodiscard]] auto w() const -> const T&
    requires ValueAtLeast<Size, 4>
  {
    return operator()(3);
  }

  /**
   * @brief Accesses an element of the vector (read-only).
   *
   * This method allows for read-only access to a specific element in the vector
   * using an index. The access pattern depends on the layout option (RowMajor
   * or ColumnMajor).
   *
   * @param index The index of the element to access.
   * @return A constant reference to the element at the specified index.
   */
  auto operator()(std::size_t index) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_(0, index);
    } else {
      return m_dataStorage_(index, 0);
    }
  }

  /**
   * @brief Accesses an element of the vector (modifiable).
   *
   * This method allows for modifying a specific element in the vector
   * using an index. The access pattern depends on the layout option (RowMajor
   * or ColumnMajor).
   *
   * @param index The index of the element to access.
   * @return A reference to the element at the specified index.
   */
  auto operator()(std::size_t index) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_(0, index);
    } else {
      return m_dataStorage_(index, 0);
    }
  }

  /**
   * @brief Returns the value of a coefficient (read-only).
   *
   * This method provides read-only access to a coefficient in the vector
   * based on the index. The access pattern depends on the layout option
   * (RowMajor or ColumnMajor).
   *
   * @param index The index of the coefficient to access.
   * @return A constant reference to the coefficient at the specified index.
   */
  [[nodiscard]] auto coeff(std::size_t index) const -> const T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_.coeff(0, index);
    } else {
      return m_dataStorage_.coeff(index, 0);
    }
  }

  /**
   * @brief Returns a reference to a coefficient (modifiable).
   *
   * This method allows for modifying a coefficient in the vector
   * based on the index. The access pattern depends on the layout option
   * (RowMajor or ColumnMajor).
   *
   * @param index The index of the coefficient to access.
   * @return A reference to the coefficient at the specified index.
   */
  auto coeffRef(std::size_t index) -> T& {
    if constexpr (Option == Options::RowMajor) {
      return m_dataStorage_.coeffRef(0, index);
    } else {
      return m_dataStorage_.coeffRef(index, 0);
    }
  }

  /**
   * @brief Returns the size of the vector (number of elements).
   *
   * @return The number of elements in the vector.
   */
  static constexpr auto GetSize() -> std::size_t { return Size; }

  /**
   * @brief Returns the size of the underlying data storage in bytes.
   *
   * This function returns the total size of the vector's underlying data
   * storage, including the storage for all elements in the vector.
   *
   * @return The size of the underlying data storage in bytes.
   */
  static constexpr auto GetDataSize() -> std::size_t {
    return UnderlyingType::GetDataSize();
  }

  /**
   * @brief Returns the storage layout option (RowMajor or ColumnMajor).
   *
   * @return The storage layout option (RowMajor or ColumnMajor).
   */
  static constexpr auto GetOption() -> Options { return Option; }

  /**
   * @brief Creates a resized copy of the current vector.
   *
   * This method creates a new vector with a different size, copying as many
   * elements as possible from the original vector. If the target size is larger
   * than the current size, the new elements are initialized to zero.
   *
   * @tparam TargetSize The size of the new vector.
   * @return A resized copy of the vector.
   */
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

  /**
   * @brief Returns a pointer to the underlying data array.
   *
   * This function provides access to the raw data of the vector, allowing for
   * low-level operations.
   *
   * @return A pointer to the underlying data array.
   */
  auto data() -> T* { return m_dataStorage_.data(); }

  /**
   * @brief Returns a constant pointer to the underlying data array.
   *
   * This function provides access to the raw data of the vector for const
   * objects, allowing for read-only operations.
   *
   * @return A constant pointer to the underlying data array.
   */
  [[nodiscard]] auto data() const -> const T* { return m_dataStorage_.data(); }

  /**
   * @brief Calculates and returns the magnitude (length) of the vector.
   *
   * @return The magnitude (length) of the vector.
   */
  auto magnitude() const -> T { return m_dataStorage_.magnitude(); }

  /**
   * @brief Calculates and returns the squared magnitude of the vector.
   *
   * This method returns the square of the vector's magnitude, which can be
   * useful in optimization to avoid calculating the square root.
   *
   * @return The squared magnitude of the vector.
   */
  auto magnitudeSquared() const -> T {
    return m_dataStorage_.magnitudeSquared();
  }

  /**
   * @brief Normalizes the vector (in-place).
   *
   * This method modifies the vector itself to make it a unit vector (i.e., with
   * a magnitude of 1).
   *
   * @note This function asserts if the magnitude of the vector is zero.
   */
  void normalize() {
    T mag = magnitude();
    assert(mag != 0
           && "Normalization error: magnitude is zero, implying a zero vector");
    *this /= mag;
  }

  /**
   * @brief Returns a normalized copy of the vector.
   *
   * This method creates a new vector that is a unit vector (i.e., with a
   * magnitude of 1), leaving the original vector unchanged.
   *
   * @return A new normalized vector.
   */
  [[nodiscard]] auto normalized() const -> Vector {
    T mag = magnitude();
    assert(mag != 0
           && "Normalization error: magnitude is zero, implying a zero vector");
    return *this / mag;
  }

  /**
   * @brief Computes the dot product of this vector and another vector.
   *
   * This function computes the dot product of two vectors using the appropriate
   * SIMD instructions for the target platform. The result is a scalar value
   * representing the dot product.
   *
   * @param other The other vector to compute the dot product with.
   * @return The dot product of the two vectors.
   */
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

  /**
   * @brief Computes the cross product of this vector and another vector.
   *
   * This function computes the cross product of two 3D vectors, returning
   * a new vector that is orthogonal to both input vectors.
   *
   * @note This function is only available for 3D vectors.
   *
   * @param other The other vector to compute the cross product with.
   * @return A new vector representing the cross product of the two vectors.
   */
  [[nodiscard]] auto cross(const Vector& other) const -> Vector
    requires ValueEqualTo<Size, 3>
  {
    Vector result;

    result.x() = this->y() * other.z() - this->z() * other.y();
    result.y() = this->z() * other.x() - this->x() * other.z();
    result.z() = this->x() * other.y() - this->y() * other.x();

    return result;
  }

  /**
   * @brief Adds two vectors element-wise.
   *
   * This operator adds two vectors element-wise, returning a new vector with
   * the result of each element-wise addition.
   *
   * @param other The vector to add.
   * @return A new vector representing the sum of the two vectors.
   */
  auto operator+(const Vector& other) const -> Vector {
    return Vector(m_dataStorage_ + other.m_dataStorage_);
  }

  /**
   * @brief Adds another vector to this vector (in-place).
   *
   * This operator adds the elements of another vector to this vector in-place.
   *
   * @param other The vector to add.
   * @return A reference to this vector after the addition.
   */
  auto operator+=(const Vector& other) -> Vector& {
    m_dataStorage_ += other.m_dataStorage_;
    return *this;
  }

  /**
   * @brief Adds a scalar to each element of the vector.
   *
   * This operator adds a scalar value to each element of the vector, returning
   * a new vector with the result.
   *
   * @param scalar The scalar value to add.
   * @return A new vector with the scalar added to each element.
   */
  auto operator+(const T& scalar) const -> Vector {
    return Vector(m_dataStorage_ + scalar);
  }

  /**
   * @brief Adds a scalar to each element of this vector (in-place).
   *
   * This operator adds a scalar value to each element of the vector in-place.
   *
   * @param scalar The scalar value to add.
   * @return A reference to this vector after the addition.
   */
  auto operator+=(const T& scalar) -> Vector& {
    m_dataStorage_ += scalar;
    return *this;
  }

  /**
   * @brief Subtracts another vector from this vector element-wise.
   *
   * This operator subtracts another vector from this vector element-wise,
   * returning a new vector with the result.
   *
   * @param other The vector to subtract.
   * @return A new vector representing the element-wise difference of the two
   * vectors.
   */
  auto operator-(const Vector& other) const -> Vector {
    return Vector(m_dataStorage_ - other.m_dataStorage_);
  }

  /**
   * @brief Subtracts another vector from this vector (in-place).
   *
   * This operator subtracts another vector from this vector element-wise,
   * updating the current vector with the result.
   *
   * @param other The vector to subtract.
   * @return A reference to this vector after the subtraction.
   */
  auto operator-=(const Vector& other) -> Vector& {
    m_dataStorage_ -= other.m_dataStorage_;
    return *this;
  }

  /**
   * @brief Subtracts a scalar from each element of the vector.
   *
   * This operator subtracts a scalar value from each element of the vector,
   * returning a new vector with the result.
   *
   * @param scalar The scalar value to subtract.
   * @return A new vector with the scalar subtracted from each element.
   */
  auto operator-(const T& scalar) const -> Vector {
    return Vector(m_dataStorage_ - scalar);
  }

  /**
   * @brief Subtracts a scalar from each element of this vector (in-place).
   *
   * This operator subtracts a scalar value from each element of the vector
   * in-place.
   *
   * @param scalar The scalar value to subtract.
   * @return A reference to this vector after the subtraction.
   */
  auto operator-=(const T& scalar) -> Vector& {
    m_dataStorage_ -= scalar;
    return *this;
  }

  /**
   * @brief Negates the vector (element-wise).
   *
   * This operator negates each element of the vector, returning a new vector
   * with the result.
   *
   * @return A new vector with each element negated.
   */
  auto operator-() const -> Vector { return Vector(-m_dataStorage_); }

  /**
   * @brief Multiplies the vector by a matrix in a row-major context (vector *
   * matrix).
   *
   * This function multiplies the vector by a matrix, returning a new vector.
   * The matrix multiplication assumes a row-major layout, meaning the vector is
   * treated as a row vector when multiplying by the matrix.
   *
   * Usage example:
   * @code
   * Vector<float, 2> vec(1.0f, 2.0f);
   * Matrix<float, 2, 3> mat = { '1, 2, 3,' '4, 5, 6' };
   * auto result = vec * mat; // result is a Vector<float, 3> { 9, 12, 15 }
   * @endcode
   *
   * @tparam Rows The number of rows in the matrix.
   * @tparam Columns The number of columns in the matrix.
   * @param matrix The matrix to multiply with.
   * @return A new vector that is the result of the vector * matrix
   * multiplication.
   * @note This function only works for row-major vectors, and the size of the
   * vector must equal the number of rows in the matrix.
   */   
  template <std::size_t Rows, std::size_t Columns>
    requires ValueEqualTo<Rows, Size> && (Option == Options::RowMajor)
  auto operator*(const Matrix<T, Rows, Columns, Option>& matrix) const
      -> Vector<T, Columns, Option> {
    return Vector<T, Columns, Option>(m_dataStorage_ * matrix);
  }

  // clang-format off

/**
 * @brief Matrix multiplication-assignment operator.
 *
 * Multiplies the current vector by the given matrix and assigns the result to the vector.
 * The matrix must be square for this operation to be valid.
 *
 * @tparam Rows The number of rows in the matrix.
 * @tparam Columns The number of columns in the matrix.
 * @param matrix The matrix to multiply with.
 * @return A reference to the current vector, now multiplied by the matrix.
 * @note This function only works when the matrices are square and have matching dimensions.
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

  /**
   * @brief Multiplies the vector by a scalar.
   *
   * This operator multiplies each element in the vector by a scalar value and
   * returns a new vector.
   *
   * @param scalar The scalar value to multiply by.
   * @return A new vector with each element multiplied by the scalar.
   */
  auto operator*(const T& scalar) const -> Vector {
    return Vector(m_dataStorage_ * scalar);
  }

  /**
   * @brief Multiplies the vector by a scalar and assigns the result to the
   * current vector.
   *
   * This operator multiplies each element in the vector by a scalar value and
   * assigns the result back to the vector.
   *
   * @param scalar The scalar value to multiply by.
   * @return A reference to the current vector, now multiplied by the scalar.
   */
  auto operator*=(const T& scalar) -> Vector& {
    m_dataStorage_ *= scalar;
    return *this;
  }

  /**
   * @brief Divides the vector by a scalar.
   *
   * This operator divides each element in the vector by a scalar value and
   * returns a new vector.
   *
   * @param scalar The scalar value to divide by.
   * @return A new vector with each element divided by the scalar.
   */
  auto operator/(const T& scalar) const -> Vector {
    return Vector(m_dataStorage_ / scalar);
  }

  /**
   * @brief Divides the vector by a scalar and assigns the result to the current
   * vector.
   *
   * This operator divides each element in the vector by a scalar value and
   * assigns the result back to the vector.
   *
   * @param scalar The scalar value to divide by.
   * @return A reference to the current vector, now divided by the scalar.
   */
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
   * Compares two vectors lexicographically. The comparison is performed element
   * by element, starting from the first element of both vectors. The comparison
   * uses the instruction set's comparison function for efficient execution.
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
   * Compares two vectors lexicographically. The comparison is performed element
   * by element, starting from the first element of both vectors. The comparison
   * uses the instruction set's comparison function for efficient execution.
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
   * Compares two vectors lexicographically. This method checks whether the
   * current vector is less than or equal to the other vector using the
   * instruction set's comparison function.
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
   * Compares two vectors lexicographically. This method checks whether the
   * current vector is greater than or equal to the other vector using the
   * instruction set's comparison function.
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

  /**
   * @brief Equality operator.
   *
   * Compares two vectors for equality. This method compares the underlying data
   * storage of both vectors to determine if they are equal.
   *
   * @param other The vector to compare against.
   * @return True if both vectors are equal, false otherwise.
   */
  auto operator==(const Vector& other) const -> bool {
    return m_dataStorage_ == other.m_dataStorage_;
  }

  /**
   * @brief Inequality operator.
   *
   * Compares two vectors for inequality. This method compares the underlying
   * data storage of both vectors to determine if they are not equal.
   *
   * @param other The vector to compare against.
   * @return True if the vectors are not equal, false otherwise.
   */
  auto operator!=(const Vector& other) const -> bool {
    return m_dataStorage_ != other.m_dataStorage_;
  }

  /**
   * @brief Output stream operator.
   *
   * Outputs the vector's elements to a stream (e.g., `std::cout`). Each element
   * is separated by a space, and the elements are printed in lexicographical
   * order.
   *
   * @param os The output stream.
   * @param vector The vector to output.
   * @return The output stream containing the vector's elements.
   */
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
   * @brief Helper class for initializing `Vector` elements using the comma
   * operator.
   *
   * This class enables initialization of `Vector` elements by chaining values
   * with the comma operator. It is especially useful for initializing vectors
   * in a concise and readable manner.
   *
   * Usage example:
   * @code
   *    Vector<float, 3> vec;
   *    vec << 1.0f, 2.0f, 3.0f; // Initializes vec to [1.0, 2.0, 3.0]
   * @endcode
   */
  class VectorInitializer {
    Vector&     m_vector_;
    std::size_t m_index_;

    public:
    /**
     * @brief Constructs a `VectorInitializer` for a given vector.
     *
     * @param vector The vector to be initialized.
     * @param index The starting index for initialization.
     */
    VectorInitializer(Vector& vector, std::size_t index)
        : m_vector_(vector)
        , m_index_(index) {}

    /**
     * @brief Adds a value to the vector at the current index.
     *
     * This operator allows for adding elements to the vector using the comma
     * operator.
     *
     * @param value The value to be added.
     * @return A reference to the current `VectorInitializer`.
     */
    auto operator,(const T& value) -> VectorInitializer& {
      if (m_index_ < Size) {
        m_vector_(m_index_++) = value;
      }
      return *this;
    }
  };

  /**
   * @brief Initializes the first element of the vector and returns a
   * `VectorInitializer`.
   *
   * This operator allows for initializing the vector using the `<<` operator,
   * followed by the comma operator.
   *
   * @param value The value to initialize the first element of the vector.
   * @return A `VectorInitializer` to continue adding values to the vector.
   */
  auto operator<<(const T& value) -> VectorInitializer {
    this->operator()(0) = value;
    return VectorInitializer(*this, 1);
  }

#else
  /**
   * @brief Initializes matrix elements using the comma operator.
   *
   * This version of the `<<` operator is used when the
   * `FEATURE_VECTOR_INITIALIZER` feature is not enabled. It forwards the
   * initialization to the underlying matrix storage.
   *
   * @param value The value to initialize the first element of the matrix.
   * @return A `MatrixInitializer` to continue adding values to the matrix.
   */
  auto operator<<(const T& value) ->
      typename Matrix<T,
                      Option == Options::RowMajor ? 1 : Size,
                      Option == Options::RowMajor ? Size : 1,
                      Option>::MatrixInitializer {
    return m_dataStorage_ << value;
  }

#endif  // FEATURE_VECTOR_INITIALIZER
  /**
   * @brief Multiplies a column-major matrix by a vector.
   *
   * This function multiplies a column-major matrix by a vector, returning a new
   * vector. The number of columns in the matrix must match the size of the
   * vector.
   *
   * Usage example:
   * @code
   *    Matrix<float, 3, 2> mat = { '1, 2', '3, 4', '5, 6' };
   *    Vector<float, 2> vec(1.0f, 2.0f);
   *    auto result = mat * vec; // result is a Vector<float, 3> { 5, 11, 17 }
   * @endcode
   *
   * @tparam T1 The type of elements in the matrix.
   * @tparam Rows1 The number of rows in the matrix.
   * @tparam Columns1 The number of columns in the matrix.
   * @tparam Option1 The layout option for the matrix (RowMajor or ColumnMajor).
   * @tparam Size1 The size of the vector.
   * @param matrix The column-major matrix.
   * @param vector The vector to multiply by the matrix.
   * @return A new vector resulting from the multiplication.
   */
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
  /**
   * @brief The underlying data storage for the vector.
   *
   * The `UnderlyingType` is defined as a matrix with either 1 row or 1 column,
   * depending on the layout option (RowMajor or ColumnMajor).
   */
  using UnderlyingType = std::conditional_t<Option == Options::RowMajor,
                                            Matrix<T, 1, Size, Option>,
                                            Matrix<T, Size, 1, Option>>;
  /**
   * @brief The underlying matrix storage for vector elements.
   *
   * This variable stores the data for the vector, which is internally
   * represented as a matrix. Depending on the layout option (RowMajor or
   * ColumnMajor), it is stored as either a row vector (1xN) or a column vector
   * (Nx1).
   */
  UnderlyingType m_dataStorage_;
};

/**
 * @brief Multiplies a matrix by a vector in a column-major context (matrix *
 * vector).
 *
 * This operator multiplies a matrix by a vector, where the matrix is in
 * column-major format and the number of columns in the matrix equals the size
 * of the vector. The result is a new vector where each element is the dot
 * product of a row in the matrix and the input vector.
 *
 * @tparam T The type of the elements in the matrix and vector.
 * @tparam Rows The number of rows in the matrix.
 * @tparam Columns The number of columns in the matrix, which must match the
 * size of the vector.
 * @tparam Option The layout option (must be column-major for this function).
 * @tparam Size The size of the vector, which must match the number of columns
 * in the matrix.
 *
 * @param matrix The input matrix (column-major format).
 * @param vector The input vector to multiply.
 * @return A new vector resulting from the multiplication.
 *
 * @note This function is only for column-major matrices where the number of
 * columns equals the size of the vector.
 *
 * Usage example:
 * @code
 * Matrix<float, 3, 2> mat = { '1, 2', '3, 4', '5, 6' };
 * Vector<float, 2> vec(1.0f, 2.0f);
 * auto result = mat * vec; // result is a Vector<float, 3> { 5, 11, 17 }
 * @endcode
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
 * @brief Performs scalar multiplication on a vector, where the scalar value is
 * the left-hand operand.
 *
 * This operator multiplies each element of the vector by a scalar value. The
 * scalar is the left-hand operand, and the vector is the right-hand operand.
 * The result is a new vector with each element scaled by the scalar.
 *
 * @tparam ScalarType The type of the scalar value.
 * @tparam T The type of the elements in the vector.
 * @tparam Size The number of elements in the vector.
 * @tparam Option The layout option (RowMajor or ColumnMajor).
 *
 * @param scalar The scalar value to multiply.
 * @param vector The vector to multiply by the scalar.
 * @return A new vector where each element is multiplied by the scalar value.
 *
 * Usage example:
 * @code
 * Vector<float, 3> vec(1.0f, 2.0f, 3.0f);
 * auto result = 2.0f * vec; // result is a Vector<float, 3> { 2.0, 4.0, 6.0 }
 * @endcode
 */
template <typename ScalarType, typename T, std::size_t Size, Options Option>
auto operator*(const ScalarType& scalar, const Vector<T, Size, Option>& vector)
    -> Vector<T, Size, Option> {
  return vector * static_cast<T>(scalar);
}

/**
 * @brief Alias for a vector of floats.
 * 
 * This defines a vector of floating point numbers with the specified size and layout.
 * 
 * @tparam Size The number of elements in the vector.
 * @tparam Option The layout option (RowMajor or ColumnMajor).
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using VectorNf = Vector<float, Size, Option>;

/**
 * @brief Alias for a vector of doubles.
 * 
 * This defines a vector of double-precision floating point numbers with the specified size and layout.
 * 
 * @tparam Size The number of elements in the vector.
 * @tparam Option The layout option (RowMajor or ColumnMajor).
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using VectorNd = Vector<double, Size, Option>;

/**
 * @brief Alias for a vector of integers.
 * 
 * This defines a vector of 32-bit signed integers with the specified size and layout.
 * 
 * @tparam Size The number of elements in the vector.
 * @tparam Option The layout option (RowMajor or ColumnMajor).
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using VectorNi = Vector<std::int32_t, Size, Option>;

/**
 * @brief Alias for a vector of unsigned integers.
 * 
 * This defines a vector of 32-bit unsigned integers with the specified size and layout.
 * 
 * @tparam Size The number of elements in the vector.
 * @tparam Option The layout option (RowMajor or ColumnMajor).
 */
template <std::size_t Size, Options Option = Options::RowMajor>
using VectorNui = Vector<std::uint32_t, Size, Option>;

/**
 * @brief Alias for a 2D vector.
 * 
 * This defines a 2D vector (vector with 2 elements) of the specified type and layout.
 * 
 * @tparam T The type of elements in the vector.
 * @tparam Option The layout option (RowMajor or ColumnMajor).
 */
template <typename T, Options Option = Options::RowMajor>
using Vector2D = Vector<T, 2, Option>;

/**
 * @brief Alias for a 3D vector.
 * 
 * This defines a 3D vector (vector with 3 elements) of the specified type and layout.
 * 
 * @tparam T The type of elements in the vector.
 * @tparam Option The layout option (RowMajor or ColumnMajor).
 */
template <typename T, Options Option = Options::RowMajor>
using Vector3D = Vector<T, 3, Option>;

/**
 * @brief Alias for a 4D vector.
 * 
 * This defines a 4D vector (vector with 4 elements) of the specified type and layout.
 * 
 * @tparam T The type of elements in the vector.
 * @tparam Option The layout option (RowMajor or ColumnMajor).
 */
template <typename T, Options Option = Options::RowMajor>
using Vector4D = Vector<T, 4, Option>;

/**
 * @brief Alias for a 2D vector of floats.
 */
using Vector2Df = Vector2D<float>;

/**
 * @brief Alias for a 3D vector of floats.
 */
using Vector3Df = Vector3D<float>;

/**
 * @brief Alias for a 4D vector of floats.
 */
using Vector4Df = Vector4D<float>;

/**
 * @brief Alias for a 2D vector of doubles.
 */
using Vector2Dd = Vector2D<double>;

/**
 * @brief Alias for a 3D vector of doubles.
 */
using Vector3Dd = Vector3D<double>;

/**
 * @brief Alias for a 4D vector of doubles.
 */
using Vector4Dd = Vector4D<double>;

/**
 * @brief Alias for a 2D vector of 32-bit signed integers.
 */
using Vector2Di = Vector2D<std::int32_t>;

/**
 * @brief Alias for a 3D vector of 32-bit signed integers.
 */
using Vector3Di = Vector3D<std::int32_t>;

/**
 * @brief Alias for a 4D vector of 32-bit signed integers.
 */
using Vector4Di = Vector4D<std::int32_t>;

/**
 * @brief Alias for a 2D vector of 32-bit unsigned integers.
 */
using Vector2Dui = Vector2D<std::uint32_t>;

/**
 * @brief Alias for a 3D vector of 32-bit unsigned integers.
 */
using Vector3Dui = Vector3D<std::uint32_t>;

/**
 * @brief Alias for a 4D vector of 32-bit unsigned integers.
 */
using Vector4Dui = Vector4D<std::uint32_t>;

}  // namespace math

#endif  // MATH_LIBRARY_VECTOR_H