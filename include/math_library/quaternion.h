/**
 * @file quaternion.h
 */

#ifndef MATH_LIBRARY_QUATERNION_H
#define MATH_LIBRARY_QUATERNION_H

#include "matrix.h"
#include "vector.h"

namespace math {

template <typename T>
class Quaternion {
  public:
  Quaternion()
      : m_data_(0, 0, 0, 1) {}

  Quaternion(const T x, const T y, const T z, const T w)
      : m_data_(x, y, z, w) {}

  explicit Quaternion(const Vector3D<T>& v, const T w)
      : m_data_(v.x(), v.y(), v.z(), w) {}

  explicit Quaternion(const Vector4D<T>& v)
      : m_data_(v) {}

  Quaternion(const Quaternion& other)
      : m_data_(other.m_data_) {}

  auto x() -> T& { return m_data_.x(); }

  auto y() -> T& { return m_data_.y(); }

  auto z() -> T& { return m_data_.z(); }

  auto w() -> T& { return m_data_.w(); }

  [[nodiscard]] auto x() const -> const T& { return m_data_.x(); }

  [[nodiscard]] auto y() const -> const T& { return m_data_.y(); }

  [[nodiscard]] auto z() const -> const T& { return m_data_.z(); }

  [[nodiscard]] auto w() const -> const T& { return m_data_.w(); }

  void setX(const T x) { m_data_.x() = x; }

  void setY(const T y) { m_data_.y() = y; }

  void setZ(const T z) { m_data_.z() = z; }

  void setW(const T w) { m_data_.w() = w; }

  auto operator+(const Quaternion& other) const -> Quaternion {
    return Quaternion(m_data_ + other.m_data_);
  }

  auto operator+=(const Quaternion& other) -> Quaternion& {
    m_data_ += other.m_data_;
    return *this;
  }

  auto operator-(const Quaternion& other) const -> Quaternion {
    return Quaternion(m_data_ - other.m_data_);
  }

  auto operator-=(const Quaternion& other) -> Quaternion& {
    m_data_ -= other.m_data_;
    return *this;
  }

  auto operator*(const Quaternion& other) const -> Quaternion {
    const T x = this->w() * other.x() + this->x() * other.w()
              + this->y() * other.z() - this->z() * other.y();
    const T y = this->w() * other.y() - this->x() * other.z()
              + this->y() * other.w() + this->z() * other.x();
    const T z = this->w() * other.z() + this->x() * other.y()
              - this->y() * other.x() + this->z() * other.w();
    const T w = this->w() * other.w() - this->x() * other.x()
              - this->y() * other.y() - this->z() * other.z();
    return Quaternion(x, y, z, w);
  }

  auto operator*(const T scalar) const -> Quaternion {
    return Quaternion(m_data_ * scalar);
  }

  auto operator*=(const T scalar) -> Quaternion& {
    m_data_ *= scalar;
    return *this;
  }

  auto operator/(const T scalar) const -> Quaternion {
    return Quaternion(m_data_ / scalar);
  }

  auto operator/=(const T scalar) -> Quaternion& {
    m_data_ /= scalar;
    return *this;
  }

  auto operator-() const -> Quaternion { return Quaternion(-m_data_); }

  /**
   * @brief Computes the conjugate of the quaternion.
   *
   * The conjugate of a quaternion is obtained by negating the vector part (x,
   * y, z) while keeping the scalar part (w) unchanged.
   *
   * @return The conjugate of the quaternion.
   */
  [[nodiscard]] auto conjugate() const -> Quaternion {
    return Quaternion(-m_data_.x(), -m_data_.y(), -m_data_.z(), m_data_.w());
  }

  /**
   * @brief Computes the inverse of the quaternion.
   *
   * The inverse of a quaternion is obtained by dividing its conjugate by the
   * square of its norm.
   *
   * @return The inverse of the quaternion.
   */
  [[nodiscard]] auto inverse() const -> Quaternion {
    const T normSquared = norm() * norm();
    return conjugate() / normSquared;
  }

  /**
   * @brief Computes the norm (magnitude) of the quaternion.
   *
   * The norm of a quaternion is the square root of the sum of the squares of
   * its components.
   *
   * @return The norm of the quaternion.
   */
  [[nodiscard]] auto norm() const -> T { return m_data_.magnitude(); }

  /**
   * @brief Normalizes the quaternion (in-place).
   *
   * @note This method modifies the quaternion itself.
   */
  void normalize() {
    T mag = norm();
    assert(
        mag != 0
        && "Normalization error: magnitude is zero, implying a zero "
           "quaternion");
    *this /= mag;
  }

  /**
   * @brief Normalizes the quaternion (non-in-place).
   *
   * @return A new normalized quaternion.
   */
  [[nodiscard]] auto normalized() const -> Quaternion {
    T mag = norm();
    assert(
        mag != 0
        && "Normalization error: magnitude is zero, implying a zero "
           "quaternion");
    return *this / mag;
  }

  /**
   * @brief Rotates a 3D vector by the quaternion.
   *
   * This method applies the rotation represented by the quaternion to the given
   * 3D vector.
   *
   * @param v The 3D vector to be rotated.
   * @return The rotated vector.
   */
  [[nodiscard]] auto rotateVector(const Vector3D<T>& v) const -> Vector3D<T> {
    const Quaternion<T> p(v.x(), v.y(), v.z(), 0);
    const Quaternion<T> rotated = *this * p * conjugate();
    return Vector3D<T>(rotated.x(), rotated.y(), rotated.z());
  }

  /**
   * @brief Converts the quaternion to a rotation matrix.
   *
   * This method computes the corresponding rotation matrix that represents the
   * same rotation as the quaternion.
   *
   * @tparam Option The storage order of the matrix (RowMajor or ColumnMajor).
   * @return The rotation matrix.
   */
  template <Options Option = Options::RowMajor>
  [[nodiscard]] auto toRotationMatrix() const -> Matrix<T, 3, 3, Option> {
    const T x = m_data_.x(), y = m_data_.y(), z = m_data_.z(), w = m_data_.w();

    if constexpr (Option == Options::RowMajor) {
      // clang-format off
        return Matrix<T, 3, 3, Option>(
            1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
            2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
            2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
        );
      // clang-format on
    } else if constexpr (Option == Options::ColumnMajor) {
      // clang-format off
        return Matrix<T, 3, 3, Option>(
            1 - 2*y*y - 2*z*z, 2*x*y + 2*z*w,     2*x*z - 2*y*w,
            2*x*y - 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z + 2*x*w,
            2*x*z + 2*y*w,     2*y*z - 2*x*w,     1 - 2*x*x - 2*y*y
        );
      // clang-format on
    }
  }

  /**
   * @brief Creates a quaternion from a rotation matrix.
   *
   * This method constructs a quaternion that represents the same rotation as
   * the given rotation matrix.
   *
   * @tparam Option The storage order of the matrix (RowMajor or ColumnMajor).
   * @param m The rotation matrix.
   * @return The quaternion representing the rotation.
   */
  template <Options Option = Options::RowMajor>
  static auto fromRotationMatrix(const Matrix<T, 3, 3, Option>& m)
      -> Quaternion {
    const T trace = m.trace();

    constexpr T kHalfInverseScaleFactor  = T(0.5);
    constexpr T kIdentityMatrixIncrement = T(1);

    if (trace > 0) {
      const T scaleFactor = kHalfInverseScaleFactor
                          / std::sqrt(trace + kIdentityMatrixIncrement);
      return fromRotationMatrixPositiveTrace(m, scaleFactor);
    } else {
      return fromRotationMatrixNegativeTrace(m);
    }
  }

  /**
   * @brief Performs spherical linear interpolation (Slerp) between two
   * quaternions.
   *
   * Slerp interpolates between two quaternions along the shortest arc on the
   * unit sphere. The interpolation parameter t should be in the range [0, 1].
   *
   * @param q1 The starting quaternion.
   * @param q2 The ending quaternion.
   * @param t The interpolation parameter.
   * @return The interpolated quaternion.
   */
  static auto slerp(const Quaternion& q1, const Quaternion& q2, const T t)
      -> Quaternion {
    const T dot   = q1.m_data_.dot(q2.m_data_);
    const T angle = std::acos(dot);
    if (std::abs(angle) < std::numeric_limits<T>::epsilon()) {
      return q1;
    }
    const T sinAngle = std::sin(angle);
    const T t1       = std::sin((1 - t) * angle) / sinAngle;
    const T t2       = std::sin(t * angle) / sinAngle;
    return Quaternion(t1 * q1.m_data_ + t2 * q2.m_data_);
  }

  /**
   * @brief Performs normalized linear interpolation (Nlerp) between two
   * quaternions.
   *
   * Nlerp interpolates between two quaternions using linear interpolation and
   * normalizes the result. The interpolation parameter t should be in the range
   * [0, 1].
   *
   * @param q1 The starting quaternion.
   * @param q2 The ending quaternion.
   * @param t The interpolation parameter.
   * @return The interpolated quaternion.
   */
  static auto nlerp(const Quaternion& q1, const Quaternion& q2, const T t)
      -> Quaternion {
    return (q1 * (1 - t) + q2 * t).normalized();
  }

  auto operator==(const Quaternion& other) const -> bool {
    return m_data_ == other.m_data_;
  }

  auto operator!=(const Quaternion& other) const -> bool {
    return m_data_ != other.m_data_;
  }

  /**
   * @brief Checks if the quaternion is approximately equal to another
   * quaternion.
   *
   * Two quaternions are considered approximately equal if the magnitude of
   * their difference is within a specified tolerance.
   *
   * @param other The quaternion to compare against.
   * @param tolerance The tolerance value for comparison (default is
   * std::numeric_limits<T>::epsilon()).
   * @return True if the quaternions are approximately equal, false otherwise.
   */
  [[nodiscard]] auto isApprox(const Quaternion& other,
                              const T           tolerance
                              = std::numeric_limits<T>::epsilon()) const
      -> bool {
    return (*this - other).norm() < tolerance;
  }

  [[nodiscard]] auto isZero() const -> bool {
    // TODO: consider non-strict comparison (since working with floating point
    // types)
    return m_data_.x() == 0 && m_data_.y() == 0 && m_data_.z() == 0
        && m_data_.w() == 0;
  }

  /**
   * @brief Computes the dot product between the quaternion and another
   * quaternion.
   *
   * The dot product of two quaternions is the sum of the products of their
   * corresponding components.
   *
   * @param other The quaternion to compute the dot product with.
   * @return The dot product of the quaternions.
   */
  [[nodiscard]] auto dot(const Quaternion& other) const -> T {
    return m_data_.dot(other.m_data_);
  }

  /**
   * @brief Computes the angle between the quaternion and another quaternion.
   *
   * The angle between two quaternions represents the rotation angle required to
   * rotate one quaternion to align with the other.
   *
   * @param other The quaternion to compute the angle with.
   * @return The angle between the quaternions in radians.
   */
  [[nodiscard]] auto angle(const Quaternion& other) const -> T {
    return std::acos(this->dot(other) / (this->norm() * other.norm())) * 2;
  }

  /**
   * @brief Computes the exponential of the quaternion.
   *
   * The exponential of a quaternion is defined as: exp(q) = exp(w) * (cos(|v|),
   * v/|v| * sin(|v|)), where w is the scalar part and v is the vector part (x,
   * y, z) of the quaternion.
   *
   * @return The exponential of the quaternion.
   */
  [[nodiscard]] auto exp() const -> Quaternion {
    // TODO: double check isZero logic (if needed to add or remove)
    if (isZero()) {
      return Quaternion(0, 0, 0, 1);
    }

    const Vector3D<T> vec   = m_data_.template resizedCopy<3>();
    const T           angle = vec.magnitude();
    const T           sina  = std::sin(angle);
    const T           cosa  = std::cos(angle);
    const Vector3D<T> axis  = vec.normalized();
    return Quaternion(axis * sina, cosa);
  }

  /**
   * @brief Computes the natural logarithm of the quaternion.
   *
   * The natural logarithm of a quaternion is defined as: log(q) = (log(|q|),
   * v/|v| * acos(w/|q|)), where w is the scalar part, v is the vector part (x,
   * y, z), and |q| is the norm of the quaternion.
   *
   * @return The natural logarithm of the quaternion.
   */
  [[nodiscard]] auto log() const -> Quaternion {
    const T           angle = std::acos(m_data_.w());
    const T           sina  = std::sin(angle);
    const Vector3D<T> vec   = m_data_.template resizedCopy<3>();
    if (std::abs(sina) < std::numeric_limits<T>::epsilon()) {
      return Quaternion(vec, 0);
    }
    const T factor = angle / sina;
    return Quaternion(vec * factor, 0);
  }

  friend auto operator<<(std::ostream& os, const Quaternion& q)
      -> std::ostream& {
    os << "(" << q.x() << ", " << q.y() << ", " << q.z() << ", " << q.w()
       << ")";
    return os;
  }

#ifdef FEATURE_QUATERNION_INITIALIZER

  /**
   * @brief Helper class for initializing Quaternion elements using the comma
   * operator (operator,).
   *
   * Usage example:
   *    Quaternion<float> q;
   *    q << 1.0f, 0.0f, 0.0f, 0.0f; // Initializes q to (1.0, 0.0, 0.0, 0.0)
   */
  class QuaternionInitializer {
    Quaternion& m_quaternion_;
    std::size_t m_index_;

    public:
    QuaternionInitializer(Quaternion& quaternion, std::size_t index)
        : m_quaternion_(quaternion)
        , m_index_(index) {}

    auto operator,(const T& value) -> QuaternionInitializer& {
      if (m_index_ < 4) {
        m_quaternion_.m_data_(m_index_++) = value;
      }
      return *this;
    }
  };

  auto operator<<(const T& value) -> QuaternionInitializer {
    m_data_.x() = value;
    return QuaternionInitializer(*this, 1);
  }

#else

  auto operator<<(const T& value) ->
      typename Matrix<T, 1, 4, Options::RowMajor>::MatrixInitializer {
    return m_data_ << value;
  }

#endif  // FEATURE_QUATERNION_INITIALIZER

  private:
  template <Options Option>
  static auto fromRotationMatrixPositiveTrace(const Matrix<T, 3, 3, Option>& m,
                                              const T scaleFactor)
      -> Quaternion {
    constexpr T kQuaternionScalar = T(0.25);

    if constexpr (Option == Options::RowMajor) {
      const T &m21 = m(2, 1), m12 = m(1, 2);
      const T &m02 = m(0, 2), m20 = m(2, 0);
      const T &m10 = m(1, 0), m01 = m(0, 1);
      return Quaternion((m21 - m12) * scaleFactor,
                        (m02 - m20) * scaleFactor,
                        (m10 - m01) * scaleFactor,
                        kQuaternionScalar / scaleFactor);
    } else if constexpr (Option == Options::ColumnMajor) {
      const T &m12 = m(1, 2), m21 = m(2, 1);
      const T &m20 = m(2, 0), m02 = m(0, 2);
      const T &m01 = m(0, 1), m10 = m(1, 0);
      return Quaternion((m12 - m21) * scaleFactor,
                        (m20 - m02) * scaleFactor,
                        (m01 - m10) * scaleFactor,
                        kQuaternionScalar / scaleFactor);
    }
  }

  template <Options Option>
  static auto fromRotationMatrixNegativeTrace(const Matrix<T, 3, 3, Option>& m)
      -> Quaternion {
    const T &m00 = m(0, 0), m11 = m(1, 1), m22 = m(2, 2);

    constexpr T kMatrixDiagonalIncrement = T(1);
    constexpr T kScaleFactor             = T(2);
    constexpr T kQuaternionScalar        = T(0.25);

    if constexpr (Option == Options::RowMajor) {
      const T &m01 = m(0, 1), m10 = m(1, 0);
      const T &m02 = m(0, 2), m20 = m(2, 0);
      const T &m12 = m(1, 2), m21 = m(2, 1);

      if (m00 > m11 && m00 > m22) {
        const T squareRootTerm
            = std::sqrt(kMatrixDiagonalIncrement + m00 - m11 - m22);
        const T denominator = kScaleFactor * squareRootTerm;
        return Quaternion(kQuaternionScalar * denominator,
                          (m01 + m10) / denominator,
                          (m02 + m20) / denominator,
                          (m21 - m12) / denominator);
      } else if (m11 > m22) {
        const T squareRootTerm
            = std::sqrt(kMatrixDiagonalIncrement + m11 - m00 - m22);
        const T denominator = kScaleFactor * squareRootTerm;
        return Quaternion((m01 + m10) / denominator,
                          kQuaternionScalar * denominator,
                          (m12 + m21) / denominator,
                          (m02 - m20) / denominator);
      } else {
        const T squareRootTerm
            = std::sqrt(kMatrixDiagonalIncrement + m22 - m00 - m11);
        const T denominator = kScaleFactor * squareRootTerm;
        return Quaternion((m02 + m20) / denominator,
                          (m12 + m21) / denominator,
                          kQuaternionScalar * denominator,
                          (m10 - m01) / denominator);
      }
    } else if constexpr (Option == Options::ColumnMajor) {
      const T &m10 = m(1, 0), m01 = m(0, 1);
      const T &m20 = m(2, 0), m02 = m(0, 2);
      const T &m21 = m(2, 1), m12 = m(1, 2);

      if (m00 > m11 && m00 > m22) {
        const T squareRootTerm
            = std::sqrt(kMatrixDiagonalIncrement + m00 - m11 - m22);
        const T denominator = kScaleFactor * squareRootTerm;
        return Quaternion(kQuaternionScalar * denominator,
                          (m10 + m01) / denominator,
                          (m20 + m02) / denominator,
                          (m12 - m21) / denominator);
      } else if (m11 > m22) {
        const T squareRootTerm
            = std::sqrt(kMatrixDiagonalIncrement + m11 - m00 - m22);
        const T denominator = kScaleFactor * squareRootTerm;
        return Quaternion((m10 + m01) / denominator,
                          kQuaternionScalar * denominator,
                          (m21 + m12) / denominator,
                          (m20 - m02) / denominator);
      } else {
        const T squareRootTerm
            = std::sqrt(kMatrixDiagonalIncrement + m22 - m00 - m11);
        const T denominator = kScaleFactor * squareRootTerm;
        return Quaternion((m20 + m02) / denominator,
                          (m21 + m12) / denominator,
                          kQuaternionScalar * denominator,
                          (m01 - m10) / denominator);
      }
    }
  }

  // m_data_ is row vector under the hood
  Vector4D<T> m_data_;
};

using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

}  // namespace math

#endif