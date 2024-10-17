/**
 * @file quaternion.h
 * @brief Implements a quaternion class for 3D rotations and other quaternion
 * operations.
 *
 * This file defines the `Quaternion` class, which supports various operations
 * such as multiplication, conjugation, normalization, and interpolation for
 * use in 3D graphics and physics simulations.
 */

#ifndef MATH_LIBRARY_QUATERNION_H
#define MATH_LIBRARY_QUATERNION_H

#include "matrix.h"
#include "vector.h"

// TODO: add copy and move assignment operator and move constructor

namespace math {

// TODO: dirty solution. Consider moving somewhere
enum class Axis {
  X = 0,
  Y = 1,
  Z = 2
};

enum class Frame {
  Static   = 0,
  Rotating = 1
};

enum class EulerRotationOrder {
  XYZ,
  XYX,
  XZY,
  XZX,
  YZX,
  YZY,
  YXZ,
  YXY,
  ZXY,
  ZXZ,
  ZYX,
  ZYZ
};

/**
 * @brief A class representing a quaternion.
 *
 * Quaternions are used to represent rotations in 3D space. This class provides
 * methods for quaternion arithmetic, rotation of vectors, conversion to/from
 * rotation matrices, and interpolation between quaternions.
 *
 * @tparam T The underlying scalar type (e.g., float, double).
 */
template <typename T>
class Quaternion {
  public:
  /**
   * @brief Default constructor that initializes the quaternion to identity (0,
   * 0, 0, 1).
   */
  Quaternion()
      : m_data_(0, 0, 0, 1) {}

  /**
   * @brief Constructor that initializes the quaternion with the given
   * components.
   *
   * @param x The x-component of the quaternion.
   * @param y The y-component of the quaternion.
   * @param z The z-component of the quaternion.
   * @param w The w-component (scalar part) of the quaternion.
   */
  Quaternion(const T x, const T y, const T z, const T w)
      : m_data_(x, y, z, w) {}

  /**
   * @brief Constructor that initializes the quaternion with a 3D vector and a
   * scalar.
   *
   * @param v The 3D vector part of the quaternion.
   * @param w The scalar part of the quaternion.
   */
  explicit Quaternion(const Vector3D<T>& v, const T w)
      : m_data_(v.x(), v.y(), v.z(), w) {}

  /**
   * @brief Constructor that initializes the quaternion with a 4D vector.
   *
   * @param v The 4D vector representing the quaternion.
   */
  explicit Quaternion(const Vector4D<T>& v)
      : m_data_(v) {}

  /**
   * @brief Copy constructor.
   *
   * @param other The quaternion to copy.
   */
  Quaternion(const Quaternion& other)
      : m_data_(other.m_data_) {}

  /**
   * @brief Access the x-component of the quaternion.
   * @return A reference to the x-component.
   */
  auto x() -> T& { return m_data_.x(); }

  /**
   * @brief Access the y-component of the quaternion.
   * @return A reference to the y-component.
   */
  auto y() -> T& { return m_data_.y(); }

  /**
   * @brief Access the z-component of the quaternion.
   * @return A reference to the z-component.
   */
  auto z() -> T& { return m_data_.z(); }

  /**
   * @brief Access the w-component of the quaternion.
   * @return A reference to the w-component.
   */
  auto w() -> T& { return m_data_.w(); }

  /**
   * @brief Access the x-component of the quaternion (const version).
   * @return A const reference to the x-component.
   */
  [[nodiscard]] auto x() const -> const T& { return m_data_.x(); }

  /**
   * @brief Access the y-component of the quaternion (const version).
   * @return A const reference to the y-component.
   */
  [[nodiscard]] auto y() const -> const T& { return m_data_.y(); }

  /**
   * @brief Access the z-component of the quaternion (const version).
   * @return A const reference to the z-component.
   */
  [[nodiscard]] auto z() const -> const T& { return m_data_.z(); }

  /**
   * @brief Access the w-component of the quaternion (const version).
   * @return A const reference to the w-component.
   */
  [[nodiscard]] auto w() const -> const T& { return m_data_.w(); }

  /**
   * @brief Sets the x-component of the quaternion.
   *
   * This method allows you to modify the x-component of the quaternion.
   *
   * @param x The new value for the x-component.
   */
  void setX(const T x) { m_data_.x() = x; }

  /**
   * @brief Sets the y-component of the quaternion.
   *
   * This method allows you to modify the y-component of the quaternion.
   *
   * @param y The new value for the y-component.
   */
  void setY(const T y) { m_data_.y() = y; }

  /**
   * @brief Sets the z-component of the quaternion.
   *
   * This method allows you to modify the z-component of the quaternion.
   *
   * @param z The new value for the z-component.
   */
  void setZ(const T z) { m_data_.z() = z; }

  /**
   * @brief Sets the w-component (scalar part) of the quaternion.
   *
   * This method allows you to modify the w-component (scalar part) of the
   * quaternion.
   *
   * @param w The new value for the w-component.
   */
  void setW(const T w) { m_data_.w() = w; }

  /**
   * @brief Adds two quaternions element-wise.
   *
   * This operator adds the components of two quaternions element-wise and
   * returns the resulting quaternion.
   *
   * @param other The quaternion to add to the current quaternion.
   * @return The resulting quaternion after addition.
   */
  auto operator+(const Quaternion& other) const -> Quaternion {
    return Quaternion(m_data_ + other.m_data_);
  }

  /**
   * @brief Adds another quaternion to the current quaternion (in-place).
   *
   * This operator adds the components of another quaternion to the current
   * quaternion element-wise and modifies the current quaternion in-place.
   *
   * @param other The quaternion to add to the current quaternion.
   * @return A reference to the modified quaternion.
   */
  auto operator+=(const Quaternion& other) -> Quaternion& {
    m_data_ += other.m_data_;
    return *this;
  }

  /**
   * @brief Subtracts one quaternion from another element-wise.
   *
   * This operator subtracts the components of the other quaternion from the
   * current quaternion element-wise and returns the resulting quaternion.
   *
   * @param other The quaternion to subtract from the current quaternion.
   * @return The resulting quaternion after subtraction.
   */
  auto operator-(const Quaternion& other) const -> Quaternion {
    return Quaternion(m_data_ - other.m_data_);
  }

  /**
   * @brief Subtracts another quaternion from the current quaternion (in-place).
   *
   * This operator subtracts the components of the other quaternion from the
   * current quaternion element-wise and modifies the current quaternion
   * in-place.
   *
   * @param other The quaternion to subtract from the current quaternion.
   * @return A reference to the modified quaternion.
   */
  auto operator-=(const Quaternion& other) -> Quaternion& {
    m_data_ -= other.m_data_;
    return *this;
  }

  /**
   * @brief Multiplies two quaternions.
   *
   * This operator multiplies the current quaternion by another quaternion and
   * returns the resulting quaternion. Quaternion multiplication combines the
   * rotations represented by the quaternions.
   *
   * @param other The quaternion to multiply with the current quaternion.
   * @return The resulting quaternion after multiplication.
   */
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

  /**
   * @brief Multiplies the quaternion by a scalar.
   *
   * This operator multiplies each component of the quaternion by a scalar
   * value.
   *
   * @param scalar The scalar value to multiply the quaternion by.
   * @return The resulting quaternion after scalar multiplication.
   */
  auto operator*(const T scalar) const -> Quaternion {
    return Quaternion(m_data_ * scalar);
  }

  /**
   * @brief Multiplies the quaternion by a scalar (in-place).
   *
   * This operator multiplies each component of the current quaternion by a
   * scalar value and modifies the quaternion in-place.
   *
   * @param scalar The scalar value to multiply the quaternion by.
   * @return A reference to the modified quaternion.
   */
  auto operator*=(const T scalar) -> Quaternion& {
    m_data_ *= scalar;
    return *this;
  }

  /**
   * @brief Divides the quaternion by a scalar.
   *
   * This operator divides each component of the quaternion by a scalar value.
   *
   * @param scalar The scalar value to divide the quaternion by.
   * @return The resulting quaternion after scalar division.
   */
  auto operator/(const T scalar) const -> Quaternion {
    return Quaternion(m_data_ / scalar);
  }

  /**
   * @brief Divides the quaternion by a scalar (in-place).
   *
   * This operator divides each component of the current quaternion by a scalar
   * value and modifies the quaternion in-place.
   *
   * @param scalar The scalar value to divide the quaternion by.
   * @return A reference to the modified quaternion.
   */
  auto operator/=(const T scalar) -> Quaternion& {
    m_data_ /= scalar;
    return *this;
  }

  /**
   * @brief Negates the quaternion.
   *
   * This operator negates each component of the quaternion, effectively
   * flipping the direction of the rotation represented by the quaternion.
   *
   * @return The negated quaternion.
   */
  auto operator-() const -> Quaternion { return Quaternion(-m_data_); }

  /**
   * @brief Computes the conjugate of the quaternion.
   *
   * The conjugate of a quaternion is obtained by negating the vector part (x,
   * y, z) while keeping the scalar part (w) unchanged. The conjugate of a
   * quaternion is useful in operations like quaternion multiplication and
   * rotations, where it can help reverse the direction of a rotation.
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
   * square of its norm. The inverse quaternion, when multiplied by the original
   * quaternion, results in the identity quaternion. It is useful for undoing or
   * reversing rotations.
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
   * its components (x, y, z, w). The norm is useful for normalizing the
   * quaternion or checking if the quaternion is a unit quaternion (one with a
   * norm of 1).
   *
   * @return The norm (magnitude) of the quaternion.
   */
  [[nodiscard]] auto norm() const -> T { return m_data_.magnitude(); }

  // TODO: add doxygen comment
  [[nodiscard]] auto squaredNorm() const -> T {
    return m_data_.magnitudeSquared();
  }

  /**
   * @brief Normalizes the quaternion (in-place).
   *
   * This method modifies the quaternion by scaling it so that its norm becomes
   * 1, making it a unit quaternion. Unit quaternions are often used in 3D
   * graphics and physics to represent rotations as they have useful
   * mathematical properties.
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
   * This method returns a new quaternion that is the normalized (unit) version
   * of the original quaternion. The original quaternion is not modified.
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
   * 3D vector using the quaternion-vector multiplication formula.
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
   * This method computes the corresponding 3x3 rotation matrix that represents
   * the same rotation as the quaternion.
   *
   * @tparam Option The storage order of the matrix (RowMajor or ColumnMajor).
   * @return A 3x3 rotation matrix that represents the quaternion's rotation.
   */
  template <Options Option = Options::RowMajor>
  [[nodiscard]] auto toRotationMatrix() const -> Matrix<T, 3, 3, Option> {
    const T x = m_data_.x(), y = m_data_.y(), z = m_data_.z(), w = m_data_.w();

    if constexpr (Option == Options::RowMajor) {
      // clang-format off
      return Matrix<T, 3, 3, Option>(
          1 - 2*y*y - 2*z*z, 2*x*y + 2*z*w,     2*x*z - 2*y*w,
          2*x*y - 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z + 2*x*w,
          2*x*z + 2*y*w,     2*y*z - 2*x*w,     1 - 2*x*x - 2*y*y
      );
      // clang-format on
    } else if constexpr (Option == Options::ColumnMajor) {
      // clang-format off
      return Matrix<T, 3, 3, Option>(
          1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w,
          2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w,
          2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y
      );
      // clang-format on
    }
  }

  // TODO: add toEulerAngles method that takes vector of angles

  // TODO: add doxygen comment
  // This additional template parameter allows for different types to be passed
  // (e.g., double format when T is float).
  template <typename U = T>
  void toEulerAngles(U&                 angle1,
                     U&                 angle2,
                     U&                 angle3,
                     EulerRotationOrder order,
                     Frame              frame = Frame::Static) {
    Axis firstAxis, secondAxis, thirdAxis;
    bool isParityOdd, isRepeated;
    getEulerOrderParameters(
        order, firstAxis, secondAxis, thirdAxis, isParityOdd, isRepeated);

    auto i = static_cast<std::size_t>(firstAxis);
    auto j = static_cast<std::size_t>(secondAxis);
    auto k = static_cast<std::size_t>(thirdAxis);

    auto norm  = squaredNorm();
    auto scale = (norm > U(0)) ? (U(2) / norm) : U(0);

    // Precompute products for the rotation matrix
    auto scaleX = x() * scale;
    auto scaleY = y() * scale;
    auto scaleZ = z() * scale;

    auto wx = w() * scaleX;
    auto wy = w() * scaleY;
    auto wz = w() * scaleZ;

    auto xx = x() * scaleX;
    auto xy = x() * scaleY;
    auto xz = x() * scaleZ;

    auto yy = y() * scaleY;
    auto yz = y() * scaleZ;
    auto zz = z() * scaleZ;

    // Convert quaternion to rotation matrix
    // clang-format off
    Matrix<U, 3, 3> R;
    R << U(1) - (yy + zz) , xy - wz          , xz + wy          ,
         xy + wz          , U(1) - (xx + zz) , yz - wx          ,
         xz - wy          , yz + wx          , U(1) - (xx + yy) ;

    const auto kSingularityThreshold = U(1e-6);

    // Extract Euler angles from the rotation matrix
    if (isRepeated) {
      auto sy = std::sqrt(R(i, j) * R(i, j) + R(i, k) * R(i, k));
      if (sy > kSingularityThreshold) {
        angle1 = std::atan2(R(i, j),  R(i, k));
        angle2 = std::atan2(sy     ,  R(i, i));
        angle3 = std::atan2(R(j, i), -R(k, i));
      } else {
        angle1 = std::atan2(-R(j, k), R(j, j));
        angle2 = std::atan2( sy     , R(i, i));
        angle3 = U(0);
      }
    } else {
      auto cy = std::sqrt(R(i, i) * R(i, i) + R(j, i) * R(j, i));
      if (cy > kSingularityThreshold) {
        angle1 = std::atan2( R(k, j), R(k, k));
        angle2 = std::atan2(-R(k, i), cy     );
        angle3 = std::atan2( R(j, i), R(i, i));
      } else {
        angle1 = std::atan2(-R(j, k), R(j, j));
        angle2 = std::atan2(-R(k, i), cy     );
        angle3 = U(0);
      }
    }
    // clang-format on

    // Handle parity
    if (isParityOdd) {
      angle1 = -angle1;
      angle2 = -angle2;
      angle3 = -angle3;
    }

    // Swap angles if frame is rotating
    if (frame == Frame::Rotating) {
      std::swap(angle1, angle3);
    }
  }

  /**
   * @brief Converts the quaternion to an axis-angle representation.
   *
   * Extracts the rotation axis and rotation angle from the quaternion.
   * The axis is returned as a normalized vector, and the angle is in radians.
   * For the identity quaternion (no rotation), the angle will be zero and the
   * axis will be (1, 0, 0).
   *
   * @param axis Output parameter to receive the rotation axis (normalized
   * vector).
   * @param angle Output parameter to receive the rotation angle in radians.
   */
  void toAxisAngle(Vector3D<T>& axis, T& angle) const {
    auto q = this->normalized();

    auto q0 = q.w();
    auto q1 = q.x();
    auto q2 = q.y();
    auto q3 = q.z();

    angle = T(2) * std::acos(q0);

    // sin(angle/2) is derived using the identity
    // sin^2(angle/2) + cos^2(angle/2) = 1,
    // therefore sin(angle/2) = sqrt(1 - cos^2(angle/2))
    auto sinHalfAngle = std::sqrt(T(1) - q0 * q0);

    if (sinHalfAngle < std::numeric_limits<T>::epsilon()) {
      // Angle is zero, so any axis will do
      axis = Vector3D<T>(T(1), T(0), T(0));
    } else {
      axis = Vector3D<T>(q1, q2, q3) / sinHalfAngle;
    }
  }

  /**
   * @brief Creates a quaternion from a rotation matrix.
   *
   * This method constructs a quaternion that represents the same rotation as
   * the given 3x3 rotation matrix.
   *
   * @tparam Option The storage order of the matrix (RowMajor or ColumnMajor).
   * @param m The 3x3 rotation matrix.
   * @return The quaternion representing the rotation.
   */
  template <Options Option = Options::RowMajor>
  static auto fromRotationMatrix(const Matrix<T, 3, 3, Option>& m)
      -> Quaternion {
    auto trace = m.trace();

    constexpr T kHalfInverseScaleFactor  = T(0.5);
    constexpr T kIdentityMatrixIncrement = T(1);

    if (trace > 0) {
      auto scaleFactor = kHalfInverseScaleFactor
                       / std::sqrt(trace + kIdentityMatrixIncrement);
      return fromRotationMatrixPositiveTrace(m, scaleFactor);
    } else {
      return fromRotationMatrixNegativeTrace(m);
    }
  }

  // TODO: add doxygen comments
  static auto fromEulerAngles(T                  angle1,
                              T                  angle2,
                              T                  angle3,
                              EulerRotationOrder order,
                              Frame frame = Frame::Static) -> Quaternion {
    Axis firstAxis, secondAxis, thirdAxis;
    bool isParityOdd, isRepeated;
    getEulerOrderParameters(
        order, firstAxis, secondAxis, thirdAxis, isParityOdd, isRepeated);

    auto i = static_cast<std::size_t>(firstAxis);
    auto j = static_cast<std::size_t>(secondAxis);
    auto k = static_cast<std::size_t>(thirdAxis);

    if (frame == Frame::Rotating) {
      std::swap(angle1, angle3);
    }
    if (isParityOdd) {
      angle2 = -angle2;
    }

    T halfA1 = angle1 * T(0.5);
    T halfA2 = angle2 * T(0.5);
    T halfA3 = angle3 * T(0.5);

    T cosA1 = std::cos(halfA1);
    T cosA2 = std::cos(halfA2);
    T cosA3 = std::cos(halfA3);

    T sinA1 = std::sin(halfA1);
    T sinA2 = std::sin(halfA2);
    T sinA3 = std::sin(halfA3);

    T cosA1_cosA3 = cosA1 * cosA3;
    T cosA1_sinA3 = cosA1 * sinA3;
    T sinA1_cosA3 = sinA1 * cosA3;
    T sinA1_sinA3 = sinA1 * sinA3;

    T qx, qy, qz, qw;

    if (isRepeated) {
      qx = cosA2 * (cosA1_sinA3 + sinA1_cosA3);
      qy = sinA2 * (cosA1_cosA3 + sinA1_sinA3);
      qz = sinA2 * (cosA1_sinA3 - sinA1_cosA3);
      qw = cosA2 * (cosA1_cosA3 - sinA1_sinA3);
    } else {
      qx = cosA2 * sinA1_cosA3 - sinA2 * cosA1_sinA3;
      qy = cosA2 * sinA1_sinA3 + sinA2 * cosA1_cosA3;
      qz = cosA2 * cosA1_sinA3 - sinA2 * sinA1_cosA3;
      qw = cosA2 * cosA1_cosA3 + sinA2 * sinA1_sinA3;
    }

    if (isParityOdd) {
      qy = -qy;
    }

    std::array<T, 3> qComponents;
    qComponents[i] = qx;
    qComponents[j] = qy;
    qComponents[k] = qz;

    return Quaternion<T>(qComponents[0], qComponents[1], qComponents[2], qw)
        .normalized();
  }

  /**
   * @brief Creates a quaternion from an axis-angle representation.
   *
   * Constructs a quaternion representing a rotation around a given axis by a
   * given angle. The axis should be a normalized vector.
   *
   * @param axis The axis of rotation (should be a normalized vector).
   * @param angle The rotation angle in radians.
   * @return The quaternion representing the rotation.
   */
  static auto fromAxisAngle(const Vector3D<T>& axis, const T angle)
      -> Quaternion {
    auto halfAngle    = angle / T(2);
    auto sinHalfAngle = std::sin(halfAngle);
    auto cosHalfAngle = std::cos(halfAngle);

    Vector3D<T> normalizedAxis = axis.normalized();

    auto q0 = cosHalfAngle;
    auto q1 = normalizedAxis.x() * sinHalfAngle;
    auto q2 = normalizedAxis.y() * sinHalfAngle;
    auto q3 = normalizedAxis.z() * sinHalfAngle;

    return Quaternion(q1, q2, q3, q0);
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
   * @param t The interpolation parameter (0.0 corresponds to q1, 1.0
   * corresponds to q2).
   * @return The interpolated quaternion.
   */
  static auto slerp(const Quaternion& q1, const Quaternion& q2, const T t)
      -> Quaternion {
    auto dot   = q1.m_data_.dot(q2.m_data_);
    auto angle = std::acos(dot);
    if (std::abs(angle) < std::numeric_limits<T>::epsilon()) {
      return q1;
    }
    auto sinAngle = std::sin(angle);
    auto t1       = std::sin((1 - t) * angle) / sinAngle;
    auto t2       = std::sin(t * angle) / sinAngle;
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
   * @param t The interpolation parameter (0.0 corresponds to q1, 1.0
   * corresponds to q2).
   * @return The interpolated and normalized quaternion.
   */
  static auto nlerp(const Quaternion& q1, const Quaternion& q2, const T t)
      -> Quaternion {
    return (q1 * (1 - t) + q2 * t).normalized();
  }

  /**
   * @brief Compares two quaternions for equality.
   *
   * Two quaternions are considered equal if all their components (x, y, z, w)
   * are exactly the same.
   *
   * @param other The quaternion to compare with.
   * @return True if the quaternions are equal, false otherwise.
   */
  auto operator==(const Quaternion& other) const -> bool {
    return m_data_ == other.m_data_;
  }

  /**
   * @brief Compares two quaternions for inequality.
   *
   * Two quaternions are considered unequal if any of their components (x, y, z,
   * w) differ.
   *
   * @param other The quaternion to compare with.
   * @return True if the quaternions are not equal, false otherwise.
   */
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

  /**
   * @brief Checks if the quaternion is zero.
   *
   * A quaternion is considered zero if all its components (x, y, z, w) are
   * zero.
   *
   * @return True if the quaternion is zero, false otherwise.
   */
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
    auto              angle = vec.magnitude();
    auto              sinA  = std::sin(angle);
    auto              cosA  = std::cos(angle);
    const Vector3D<T> axis  = vec.normalized();
    return Quaternion(axis * sinA, cosA);
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
    auto              angle = std::acos(m_data_.w());
    auto              sinA  = std::sin(angle);
    const Vector3D<T> vec   = m_data_.template resizedCopy<3>();
    if (std::abs(sinA) < std::numeric_limits<T>::epsilon()) {
      return Quaternion(vec, 0);
    }
    auto factor = angle / sinA;
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
   * This class allows for intuitive initialization of `Quaternion` elements by
   * using the comma operator. Each element of the quaternion can be set in
   * sequence through multiple uses of the comma operator.
   *
   * Usage example:
   * @code
   * Quaternion<float> q;
   * q << 1.0f, 0.0f, 0.0f, 0.0f; // Initializes q to (1.0, 0.0, 0.0, 0.0)
   * @endcode
   */
  class QuaternionInitializer {
    Quaternion& m_quaternion_;
    std::size_t m_index_;

    public:
    /**
     * @brief Constructs a QuaternionInitializer.
     *
     * @param quaternion Reference to the quaternion being initialized.
     * @param index The index of the element to initialize.
     */
    QuaternionInitializer(Quaternion& quaternion, std::size_t index)
        : m_quaternion_(quaternion)
        , m_index_(index) {}

    /**
     * @brief Comma operator for sequentially initializing quaternion elements.
     *
     * This operator allows sequential initialization of the quaternion
     * elements. It will stop initializing elements once all four components are
     * set.
     *
     * @param value The value to assign to the current element.
     * @return A reference to the QuaternionInitializer for further
     * initialization.
     */
    auto operator,(const T& value) -> QuaternionInitializer& {
      if (m_index_ < 4) {
        m_quaternion_.m_data_(m_index_++) = value;
      }
      return *this;
    }
  };

  /**
   * @brief Initializes the quaternion with the first element using the `<<`
   * operator.
   *
   * This operator sets the x-component of the quaternion and returns a
   * `QuaternionInitializer` object for further element initialization.
   *
   * @param value The value to assign to the x-component.
   * @return A QuaternionInitializer to initialize the remaining elements.
   */
  auto operator<<(const T& value) -> QuaternionInitializer {
    m_data_.x() = value;
    return QuaternionInitializer(*this, 1);
  }

#else

  /**
   * @brief Initializes the quaternion using the Matrix initializer when
   * FEATURE_QUATERNION_INITIALIZER is disabled.
   *
   * This operator forwards the initialization to the `MatrixInitializer` when
   * FEATURE_QUATERNION_INITIALIZER is not enabled.
   *
   * @param value The value to assign to the x-component.
   * @return A MatrixInitializer to initialize the remaining elements.
   */
  auto operator<<(const T& value) ->
      typename Matrix<T, 1, 4, Options::RowMajor>::MatrixInitializer {
    return m_data_ << value;
  }

#endif  // FEATURE_QUATERNION_INITIALIZER

  private:
  // TODO: add doxygen comment
  static void getEulerOrderParameters(EulerRotationOrder order,
                                      Axis&              firstAxis,
                                      Axis&              secondAxis,
                                      Axis&              thirdAxis,
                                      bool&              isParityOdd,
                                      bool&              hasRepetition) {
    switch (order) {
      case EulerRotationOrder::XYZ:
        firstAxis     = Axis::X;
        isParityOdd   = false;
        hasRepetition = false;
        break;
      case EulerRotationOrder::XYX:
        firstAxis     = Axis::X;
        isParityOdd   = false;
        hasRepetition = true;
        break;
      case EulerRotationOrder::XZY:
        firstAxis     = Axis::X;
        isParityOdd   = true;
        hasRepetition = false;
        break;
      case EulerRotationOrder::XZX:
        firstAxis     = Axis::X;
        isParityOdd   = true;
        hasRepetition = true;
        break;
      case EulerRotationOrder::YZX:
        firstAxis     = Axis::Y;
        isParityOdd   = false;
        hasRepetition = false;
        break;
      case EulerRotationOrder::YZY:
        firstAxis     = Axis::Y;
        isParityOdd   = false;
        hasRepetition = true;
        break;
      case EulerRotationOrder::YXZ:
        firstAxis     = Axis::Y;
        isParityOdd   = true;
        hasRepetition = false;
        break;
      case EulerRotationOrder::YXY:
        firstAxis     = Axis::Y;
        isParityOdd   = true;
        hasRepetition = true;
        break;
      case EulerRotationOrder::ZXY:
        firstAxis     = Axis::Z;
        isParityOdd   = false;
        hasRepetition = false;
        break;
      case EulerRotationOrder::ZXZ:
        firstAxis     = Axis::Z;
        isParityOdd   = false;
        hasRepetition = true;
        break;
      case EulerRotationOrder::ZYX:
        firstAxis     = Axis::Z;
        isParityOdd   = true;
        hasRepetition = false;
        break;
      case EulerRotationOrder::ZYZ:
        firstAxis     = Axis::Z;
        isParityOdd   = true;
        hasRepetition = true;
        break;
      default:
        assert(false && "Invalid Euler rotation order");
        break;
    }

    int32_t firstAxisIndex = static_cast<int32_t>(firstAxis);
    int32_t secondAxisIndex;
    int32_t thirdAxisIndex;

    // Determine the second and third axes based on parity
    if (!isParityOdd) {
      secondAxisIndex = (firstAxisIndex + 1) % 3;
      thirdAxisIndex  = (firstAxisIndex + 2) % 3;
    } else {
      secondAxisIndex = (firstAxisIndex + 2) % 3;
      thirdAxisIndex  = (firstAxisIndex + 1) % 3;
    }

    secondAxis = static_cast<Axis>(secondAxisIndex);
    thirdAxis  = static_cast<Axis>(thirdAxisIndex);
  }

  /**
   * @brief Constructs a quaternion from a rotation matrix when the trace is
   * positive.
   *
   * This function creates a quaternion from a rotation matrix when the trace
   * of the matrix is positive. It uses the appropriate scale factor to compute
   * the quaternion components.
   *
   * @tparam Option The storage order of the matrix (RowMajor or ColumnMajor).
   * @param m The rotation matrix.
   * @param scaleFactor The scale factor used to compute the quaternion
   * components.
   * @return The quaternion representing the rotation.
   */
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

  /**
   * @brief Constructs a quaternion from a rotation matrix when the trace is
   * negative.
   *
   * This function creates a quaternion from a rotation matrix when the trace
   * of the matrix is negative. It chooses the appropriate quaternion component
   * based on the largest diagonal element of the matrix.
   *
   * @tparam Option The storage order of the matrix (RowMajor or ColumnMajor).
   * @param m The rotation matrix.
   * @return The quaternion representing the rotation.
   */
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

  /**
   * @brief Internal data representing the quaternion components.
   *
   * The quaternion is stored as a 4D vector (x, y, z, w), where (x, y, z)
   * represent the vector part and w represents the scalar part of the
   * quaternion.
   *
   * @note The data is stored as a row vector internally.
   */
  Vector4D<T> m_data_;
};

/**
 * @brief Typedef for a quaternion with float components.
 *
 * This type is used when working with quaternions that use `float` as the
 * scalar type.
 */
using Quaternionf = Quaternion<float>;

/**
 * @brief Typedef for a quaternion with double components.
 *
 * This type is used when working with quaternions that use `double` as the
 * scalar type.
 */
using Quaterniond = Quaternion<double>;

}  // namespace math

#endif  // MATH_LIBRARY_QUATERNION_H
