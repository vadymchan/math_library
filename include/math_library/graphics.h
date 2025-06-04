/**
 * @file graphics.h
 * @brief Header file for matrix transformations and projection functions in a
 * 3D space.
 *
 * This file contains a variety of utility functions for generating and
 * manipulating 4x4 transformation matrices for 3D graphics. The file includes
 * functions to create translation, scaling, and rotation matrices, as well as
 * view and projection matrices for both left-handed and right-handed coordinate
 * systems. It also includes functions for transforming points and vectors, as
 * well as some common vector utilities.
 *
 * Supported matrix operations include:
 * - Translation matrices.
 * - Scaling matrices.
 * - Rotation matrices (around X, Y, and Z axes, and arbitrary axes).
 * - View matrices (look-at, look-to).
 * - Perspective projection matrices (both finite and infinite far plane).
 * - Orthographic projection matrices.
 * - Frustum projection matrices.
 *
 * The functions support both row-major and column-major matrix layouts through
 * a template parameter.
 *
 */

#ifndef MATH_LIBRARY_GRAPHICS_H
#define MATH_LIBRARY_GRAPHICS_H

#include "matrix.h"
#include "point.h"
#include "vector.h"

// TODO:
// - for matrix initialization, use << operator instead of operator(row,
// column);
// - add additional functions that will take matrix as an argument and use it as
// a return value (to remove things like math::g_someFunction<float,
// math::Options::ColumnMajor>(...)
// - add general recomendations (what functions to use and how)

namespace math {

/**
 * @brief Creates a translation matrix.
 *
 * Generates a 4x4 translation matrix for translating objects in 3D space by the
 * specified offsets along the x, y, and z axes.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param dx Translation offset along the x-axis.
 * @param dy Translation offset along the y-axis.
 * @param dz Translation offset along the z-axis.
 * @return A 4x4 translation matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_translate(T dx, T dy, T dz) -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> translateMat{T()};
  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    translateMat <<
      1,   0,   0,   0,
      0,   1,   0,   0,
      0,   0,   1,   0,
      dx,  dy,  dz,  1;
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    translateMat <<
      1,   0,   0,   dx,
      0,   1,   0,   dy,
      0,   0,   1,   dz,
      0,   0,   0,   1;
    // clang-format on
  }
  return translateMat;
}

/**
 * @brief Creates a translation matrix.
 *
 * Generates a 4x4 translation matrix for translating objects in 3D space by the
 * specified translation vector.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param translation A vector specifying the translation offsets along the x,
 * y, and z axes.
 * @return A 4x4 translation matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_translate(const Vector<T, 3, Option>& translation)
    -> Matrix<T, 4, 4, Option> {
  return g_translate<T, Option>(
      translation.x(), translation.y(), translation.z());
}

/**
 * @brief Adds a translation to an existing transformation matrix.
 *
 * Modifies the given 4x4 transformation matrix by adding a translation
 * specified by the offsets along the x, y, and z axes.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param matrix The transformation matrix to modify.
 * @param dx Translation offset along the x-axis.
 * @param dy Translation offset along the y-axis.
 * @param dz Translation offset along the z-axis.
 */
template <typename T, Options Option>
void g_addTranslate(Matrix<T, 4, 4, Option>& matrix, T dx, T dy, T dz) {
  if constexpr (Option == Options::RowMajor) {
    matrix(3, 0) += dx;
    matrix(3, 1) += dy;
    matrix(3, 2) += dz;
  } else if constexpr (Option == Options::ColumnMajor) {
    matrix(0, 3) += dx;
    matrix(1, 3) += dy;
    matrix(2, 3) += dz;
  }
}

/**
 * @brief Adds a translation to an existing transformation matrix.
 *
 * Modifies the given 4x4 transformation matrix by adding a translation
 * specified by a translation vector.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param matrix The transformation matrix to modify.
 * @param translation A vector specifying the translation offsets along the x,
 * y, and z axes.
 */
template <typename T, Options Option>
void g_addTranslate(Matrix<T, 4, 4, Option>&    matrix,
                    const Vector<T, 3, Option>& translation) {
  g_addTranslate(matrix, translation.x(), translation.y(), translation.z());
}

/**
 * @brief Sets the translation component of a transformation matrix.
 *
 * Updates the translation component of the given 4x4 transformation matrix to
 * the specified offsets along the x, y, and z axes.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param matrix The transformation matrix to modify.
 * @param dx Translation offset along the x-axis.
 * @param dy Translation offset along the y-axis.
 * @param dz Translation offset along the z-axis.
 */
template <typename T, Options Option>
void g_setTranslate(Matrix<T, 4, 4, Option>& matrix, T dx, T dy, T dz) {
  Vector<T, 4, Option> translation(dx, dy, dz, matrix(3, 3));
  if constexpr (Option == Options::RowMajor) {
    matrix.setRow<3>(translation);
  } else if constexpr (Option == Options::ColumnMajor) {
    matrix.setColumn<3>(translation);
  }
}

/**
 * @brief Sets the translation component of a transformation matrix.
 *
 * Updates the translation component of the given 4x4 transformation matrix to
 * the specified translation vector.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param matrix The transformation matrix to modify.
 * @param translation A vector specifying the translation offsets along the x,
 * y, and z axes.
 */
template <typename T, Options Option>
void g_setTranslate(Matrix<T, 4, 4, Option>&    matrix,
                    const Vector<T, 3, Option>& translation) {
  g_setTranslate(matrix, translation.x(), translation.y(), translation.z());
}

/**
 * @brief Creates a scaling matrix.
 *
 * Generates a 4x4 scaling matrix for scaling objects in 3D space by the
 * specified factors along the x, y, and z axes.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param sx Scaling factor along the x-axis.
 * @param sy Scaling factor along the y-axis.
 * @param sz Scaling factor along the z-axis.
 * @return A 4x4 scaling matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_scale(T sx, T sy, T sz) -> Matrix<T, 4, 4, Option> {
  // clang-format off
  return Matrix<T, 4, 4, Option>{
    sx,   0,    0,    0, 
    0,    sy,   0,    0,
    0,    0,    sz,   0,
    0,    0,    0,    1};
  // clang-format on
}

/**
 * @brief Creates a scaling matrix.
 *
 * Generates a 4x4 scaling matrix for scaling objects in 3D space by the
 * specified scaling vector.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param scale A vector specifying the scaling factors along the x, y, and z
 * axes.
 * @return A 4x4 scaling matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_scale(const Vector<T, 3, Option>& scale) -> Matrix<T, 4, 4, Option> {
  return g_scale<T, Option>(scale.x(), scale.y(), scale.z());
}

// BEGIN: rotation matrix creation functions
// ----------------------------------------------------------------------------

/**
 * @brief Creates a rotation matrix around the X-axis in a right-handed
 * coordinate system.
 *
 * Generates a 4x4 rotation matrix that rotates objects around the X-axis by the
 * specified angle (in radians).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angle The rotation angle around the X-axis in radians.
 * @return A 4x4 rotation matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateRhX(T angle) -> Matrix<T, 4, 4, Option> {
  const T kCosAngle = std::cos(angle);
  const T kSinAngle = std::sin(angle);

  Matrix<T, 4, 4, Option> rotateMat{T()};

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    rotateMat << 1,   0,           0,          0,
                 0,   kCosAngle,   kSinAngle,  0,
                 0,  -kSinAngle,   kCosAngle,  0,
                 0,   0,           0,          1;
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    rotateMat << 1,   0,           0,          0,
                 0,   kCosAngle,  -kSinAngle,  0,
                 0,   kSinAngle,   kCosAngle,  0,
                 0,   0,           0,          1;
    // clang-format on
  }
  return rotateMat;
}

/**
 * @brief Creates a rotation matrix around the Y-axis in a right-handed
 * coordinate system.
 *
 * Generates a 4x4 rotation matrix that rotates objects around the Y-axis by the
 * specified angle (in radians).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angle The rotation angle around the Y-axis in radians.
 * @return A 4x4 rotation matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateRhY(T angle) -> Matrix<T, 4, 4, Option> {
  const T kCosAngle = std::cos(angle);
  const T kSinAngle = std::sin(angle);

  Matrix<T, 4, 4, Option> rotateMat{T()};

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    rotateMat <<  kCosAngle,   0,  -kSinAngle,  0,
                  0,           1,   0,          0,
                  kSinAngle,   0,   kCosAngle,  0,
                  0,           0,   0,          1;
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    rotateMat <<  kCosAngle,   0,   kSinAngle,  0,
                  0,           1,   0,          0,
                 -kSinAngle,   0,   kCosAngle,  0,
                  0,           0,   0,          1;
    // clang-format on
  }
  return rotateMat;
}

/**
 * @brief Creates a rotation matrix around the Z-axis in a right-handed
 * coordinate system.
 *
 * Generates a 4x4 rotation matrix that rotates objects around the Z-axis by the
 * specified angle (in radians).
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angle The rotation angle around the Z-axis in radians.
 * @return A 4x4 rotation matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateRhZ(T angle) -> Matrix<T, 4, 4, Option> {
  const T kCosAngle = std::cos(angle);
  const T kSinAngle = std::sin(angle);

  Matrix<T, 4, 4, Option> rotateMat{T()};

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    rotateMat <<  kCosAngle,   kSinAngle,  0,  0,
                 -kSinAngle,   kCosAngle,  0,  0,
                  0,           0,          1,  0,
                  0,           0,          0,  1;
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    rotateMat <<  kCosAngle,  -kSinAngle,  0,  0,
                  kSinAngle,   kCosAngle,  0,  0,
                  0,           0,          1,  0,
                  0,           0,          0,  1;
    // clang-format on
  }
  return rotateMat;
}

/**
 * @brief Creates a rotation matrix in the right-handed coordinate system.
 *
 * This function generates a 4x4 rotation matrix that represents the combined
 * rotation around the X, Y, and Z axes. The rotation order applied is
 * Z (roll) -> X (pitch) -> Y (yaw)
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angleX The rotation angle around the X-axis (roll) in radians.
 * @param angleY The rotation angle around the Y-axis (pitch) in radians.
 * @param angleZ The rotation angle around the Z-axis (yaw) in radians.
 *
 * @return A 4x4 rotation matrix in the right-handed coordinate system.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateRh(T angleX, T angleY, T angleZ) -> Matrix<T, 4, 4, Option> {
  const T kSX = std::sin(angleX);
  const T kCX = std::cos(angleX);
  const T kSY = std::sin(angleY);
  const T kCY = std::cos(angleY);
  const T kSZ = std::sin(angleZ);
  const T kCZ = std::cos(angleZ);

  Matrix<T, 4, 4, Option> rotateMat{T()};

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    rotateMat <<
      kCY * kCZ + kSY * kSX * kSZ,    kCX * kSZ,    kCY * kSX * kSZ - kSY * kCZ,   0,
      kCZ * kSY * kSX - kCY * kSZ,    kCX * kCZ,    kSY * kSZ + kCY * kSX * kCZ,   0,
      kCX * kSY,                     -kSX,          kCY * kCX,                     0,
      0,                              0,            0,                             1;
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    rotateMat <<
      kCY * kCZ + kSY * kSX * kSZ,    kCZ * kSY * kSX - kCY * kSZ,    kCX * kSY,   0,
      kCX * kSZ,                      kCX * kCZ,                     -kSX,         0,
      kCY * kSX * kSZ - kSY * kCZ,    kSY * kSZ + kCY * kSX * kCZ,    kCY * kCX,   0,
      0,                              0,                              0,           1;
    // clang-format on
  }
  return rotateMat;
}

/**
 * @brief Creates a rotation matrix in the right-handed coordinate system.
 *
 * Generates a 4x4 rotation matrix that represents the combined rotation around
 * the X, Y, and Z axes.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angles A vector containing rotation angles around the X, Y, and Z axes
 * (in radians).
 * @return A 4x4 rotation matrix in the right-handed coordinate system.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateRh(const Vector<T, 3, Option>& angles) -> Matrix<T, 4, 4, Option> {
  return g_rotateRh<T, Option>(angles.x(), angles.y(), angles.z());
}

/**
 * @brief Creates a rotation matrix for rotation around an arbitrary axis in a
 * right-handed coordinate system.
 *
 * Utilizes Rodrigues' rotation formula to generate a 4x4 rotation matrix given
 * an arbitrary axis and rotation angle. The axis does not need to be normalized
 * as the function will normalize it.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param axis The 3D vector representing the axis of rotation.
 * @param angle The rotation angle around the axis, in radians.
 *
 * @note This function is designed for right-handed coordinate systems. It
 * automatically normalizes the axis of rotation.
 *
 * @return A 4x4 rotation matrix representing rotation around the specified
 * axis.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateRh(const Vector<T, 3, Option>& axis, T angle)
    -> Matrix<T, 4, 4, Option> {
  const T kCosAngle    = std::cos(angle);
  const T kSinAngle    = std::sin(angle);
  const T kOneMinusCos = 1 - kCosAngle;

  auto normalizedAxis = axis.normalized();

  const T& x = normalizedAxis.x();
  const T& y = normalizedAxis.y();
  const T& z = normalizedAxis.z();

  Matrix<T, 4, 4, Option> rotateMat{T()};

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    rotateMat <<
      kCosAngle  +  x*x*kOneMinusCos,   x*y*kOneMinusCos - z*kSinAngle,   x*z*kOneMinusCos + y*kSinAngle,   0,              
      y*x*kOneMinusCos + z*kSinAngle,   kCosAngle  +  y*y*kOneMinusCos,   y*z*kOneMinusCos - x*kSinAngle,   0,
      z*x*kOneMinusCos - y*kSinAngle,   z*y*kOneMinusCos + x*kSinAngle,   kCosAngle  +  z*z*kOneMinusCos,   0, 
      0,                                0,                                0,                                1;
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    rotateMat <<
      kCosAngle  +  x*x*kOneMinusCos,   y*x*kOneMinusCos + z*kSinAngle,   z*x*kOneMinusCos - y*kSinAngle,   0, 
      x*y*kOneMinusCos - z*kSinAngle,   kCosAngle  +  y*y*kOneMinusCos,   z*y*kOneMinusCos + x*kSinAngle,   0,
      x*z*kOneMinusCos + y*kSinAngle,   y*z*kOneMinusCos - x*kSinAngle,   kCosAngle  +  z*z*kOneMinusCos,   0, 
      0,                                0,                                0,                                1;
    // clang-format on
  }
  return rotateMat;
}

/**
 * @brief Creates a rotation matrix around the X-axis in a left-handed
 * coordinate system.
 *
 * Generates a 4x4 rotation matrix that rotates objects around the X-axis by the
 * specified angle (in radians). This function adapts the right-handed rotation
 * function by inverting the angle.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angle The rotation angle around the X-axis in radians.
 * @return A 4x4 rotation matrix in a left-handed coordinate system.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateLhX(T angle) -> Matrix<T, 4, 4, Option> {
  return g_rotateRhX<T, Option>(-angle);
}

/**
 * @brief Creates a rotation matrix around the Y-axis in a left-handed
 * coordinate system.
 *
 * Generates a 4x4 rotation matrix that rotates objects around the Y-axis by the
 * specified angle (in radians). This function adapts the right-handed rotation
 * function by inverting the angle.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angle The rotation angle around the Y-axis in radians.
 * @return A 4x4 rotation matrix in a left-handed coordinate system.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateLhY(T angle) -> Matrix<T, 4, 4, Option> {
  return g_rotateRhY<T, Option>(-angle);
}

/**
 * @brief Creates a rotation matrix around the Z-axis in a left-handed
 * coordinate system.
 *
 * Generates a 4x4 rotation matrix that rotates objects around the Z-axis by the
 * specified angle (in radians). This function adapts the right-handed rotation
 * function by inverting the angle.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angle The rotation angle around the Z-axis in radians.
 * @return A 4x4 rotation matrix in a left-handed coordinate system.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateLhZ(T angle) -> Matrix<T, 4, 4, Option> {
  return g_rotateRhZ<T, Option>(-angle);
}

/**
 * @brief Creates a combined rotation matrix around X, Y, and Z axes in the
 * left-handed coordinate system.
 *
 * This function generates a 4x4 rotation matrix that represents the combined
 * rotation around the X, Y, and Z axes (pitch, yaw, and roll) in the order of
 * Z (roll) -> X (pitch) -> Y (yaw) using the right-handed function but inverts
 * the angles for left-handed coordinate system adaptation.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angleX The rotation angle around the X-axis (pitch) in radians.
 * @param angleY The rotation angle around the Y-axis (yaw) in radians.
 * @param angleZ The rotation angle around the Z-axis (roll) in radians.
 *
 * @return A 4x4 rotation matrix that operates in the left-handed coordinate
 * system.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateLh(T angleX, T angleY, T angleZ) -> Matrix<T, 4, 4, Option> {
  return g_rotateRh<T, Option>(-angleX, -angleY, -angleZ);
}

/**
 * @brief Creates a rotation matrix in the left-handed coordinate system.
 *
 * Generates a 4x4 rotation matrix that represents the combined rotation around
 * the X, Y, and Z axes.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param angles A vector containing rotation angles around the X, Y, and Z axes
 * (in radians).
 * @return A 4x4 rotation matrix in the left-handed coordinate system.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateLh(const Vector<T, 3, Option>& angles) -> Matrix<T, 4, 4, Option> {
  return g_rotateLh<T, Option>(angles.x(), angles.y(), angles.z());
}

/**
 * @brief Creates a rotation matrix for rotation around an arbitrary axis in a
 * left-handed coordinate system.
 *
 * Utilizes Rodrigues' rotation formula to generate a 4x4 rotation matrix given
 * an arbitrary axis and rotation angle. The function inverts the angle for
 * adaptation to left-handed coordinate systems but maintains the axis
 * direction.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param axis The 3D vector representing the axis of rotation.
 * @param angle The rotation angle around the axis, in radians.
 *
 * @note This function normalizes the axis of rotation automatically.
 *
 * @return A 4x4 rotation matrix representing rotation around the specified
 * axis.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateLh(const Vector<T, 3, Option>& axis, T angle)
    -> Matrix<T, 4, 4, Option> {
  return g_rotateRh<T, Option>(axis, -angle);
}

// END: rotation matrix creation functions
// ----------------------------------------------------------------------------

// BEGIN: view matrix creation functions
// ----------------------------------------------------------------------------

/**
 * @brief Creates a right-handed view matrix using an eye point, a target point,
 * and an up vector.
 *
 * Generates a 4x4 view matrix that transforms world coordinates to view
 * coordinates, using the specified eye position, target position, and up vector
 * in a right-handed coordinate system.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param eye The position of the camera (eye point) in world space.
 * @param target The target point the camera is looking at in world space.
 * @param worldUp The world's up vector.
 * @return A 4x4 view matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_lookAtRh(const Vector3<T, Option>& eye,
                const Vector3<T, Option>& target,
                const Vector3<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
  auto f = (target - eye).normalized();
  auto r = worldUp.cross(f).normalized();
  auto u = f.cross(r);

  Matrix<T, 4, 4, Option> viewMatrix;

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    viewMatrix <<
       r.x(),        u.x(),       -f.x(),       T(),
       r.y(),        u.y(),       -f.y(),       T(),
       r.z(),        u.z(),       -f.z(),       T(),
      -eye.dot(r),  -eye.dot(u),  -eye.dot(f),  T(1);
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    viewMatrix <<
       r.x(),   r.y(),   r.z(),  -eye.dot(r),
       u.x(),   u.y(),   u.z(),  -eye.dot(u),
      -f.x(),  -f.y(),  -f.z(),  -eye.dot(f),
       T(),     T(),     T(),     T(1);
    // clang-format on
  }

  return viewMatrix;
}

/**
 * @brief Creates a left-handed view matrix using an eye point, a target point,
 * and an up vector.
 *
 * Generates a 4x4 view matrix that transforms world coordinates to view
 * coordinates, using the specified eye position, target position, and up vector
 * in a left-handed coordinate system.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param eye The position of the camera (eye point) in world space.
 * @param target The target point the camera is looking at in world space.
 * @param worldUp The world's up vector.
 * @return A 4x4 view matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_lookAtLh(const Vector3<T, Option>& eye,
                const Vector3<T, Option>& target,
                const Vector3<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
  auto f = (target - eye).normalized();
  auto r = worldUp.cross(f).normalized();
  auto u = f.cross(r);

  Matrix<T, 4, 4, Option> viewMatrix;

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    viewMatrix <<
       r.x(),        u.x(),        f.x(),       T(),
       r.y(),        u.y(),        f.y(),       T(),
       r.z(),        u.z(),        f.z(),       T(),
      -eye.dot(r),  -eye.dot(u),  -eye.dot(f),  T(1);
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    viewMatrix <<
      r.x(),  r.y(),  r.z(),  -eye.dot(r),
      u.x(),  u.y(),  u.z(),  -eye.dot(u),
      f.x(),  f.y(),  f.z(),  -eye.dot(f),
      T(),    T(),    T(),     T(1);
    // clang-format on
  }

  return viewMatrix;
}

/**
 * @brief Creates a right-handed view matrix using an eye point, a view
 * direction, and an up vector.
 *
 * Generates a 4x4 view matrix that transforms world coordinates to view
 * coordinates, using the specified eye position, view direction, and up vector
 * in a right-handed coordinate system.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param eye The position of the camera (eye point) in world space.
 * @param direction The viewing direction of the camera.
 * @param worldUp The world's up vector.
 * @return A 4x4 view matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_lookToRh(const Vector3<T, Option>& eye,
                const Vector3<T, Option>& direction,
                const Vector3<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
  auto f = direction.normalized();
  auto r = worldUp.cross(f).normalized();
  auto u = f.cross(r);

  Matrix<T, 4, 4, Option> viewMatrix;

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    viewMatrix <<
       r.x(),        u.x(),       -f.x(),       T(),
       r.y(),        u.y(),       -f.y(),       T(),
       r.z(),        u.z(),       -f.z(),       T(),
      -eye.dot(r),  -eye.dot(u),  -eye.dot(f),  T(1);
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    viewMatrix <<
       r.x(),   r.y(),   r.z(),  -eye.dot(r),
       u.x(),   u.y(),   u.z(),  -eye.dot(u),
      -f.x(),  -f.y(),  -f.z(),  -eye.dot(f),
       T(),     T(),     T(),     T(1);
    // clang-format on
  }

  return viewMatrix;
}

/**
 * @brief Creates a left-handed view matrix using an eye point, a view
 * direction, and an up vector.
 *
 * Generates a 4x4 view matrix that transforms world coordinates to view
 * coordinates, using the specified eye position, view direction, and up vector
 * in a left-handed coordinate system.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param eye The position of the camera (eye point) in world space.
 * @param direction The viewing direction of the camera.
 * @param worldUp The world's up vector.
 * @return A 4x4 view matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_lookToLh(const Vector3<T, Option>& eye,
                const Vector3<T, Option>& direction,
                const Vector3<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
  auto f = direction.normalized();
  auto r = worldUp.cross(f).normalized();
  auto u = f.cross(r);

  Matrix<T, 4, 4, Option> viewMatrix;

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    viewMatrix <<
       r.x(),        u.x(),        f.x(),       T(),
       r.y(),        u.y(),        f.y(),       T(),
       r.z(),        u.z(),        f.z(),       T(),
      -eye.dot(r),  -eye.dot(u),  -eye.dot(f),  T(1);
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    viewMatrix <<
      r.x(),  r.y(),  r.z(),  -eye.dot(r),
      u.x(),  u.y(),  u.z(),  -eye.dot(u),
      f.x(),  f.y(),  f.z(),  -eye.dot(f),
      T(),    T(),    T(),     T(1);
    // clang-format on
  }

  return viewMatrix;
}

// END: view matrix creation functions
// ----------------------------------------------------------------------------

// BEGIN: perspective projection creation matrix
// ----------------------------------------------------------------------------

/**
 * @brief Generates a right-handed perspective projection matrix with a depth
 * range of negative one to one.
 *
 * Creates a 4x4 perspective projection matrix using the specified field of
 * view, aspect ratio, and near and far clipping planes.
 *
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param aspect Aspect ratio, defined as view space width divided by height.
 * @param nearZ Near clipping plane distance.
 * @param farZ Far clipping plane distance.
 * @return A 4x4 perspective projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhNo(T fovY, T aspect, T nearZ, T farZ)
    -> Matrix<T, 4, 4, Option> {
  // Validate aspect ratio to prevent division by zero
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspeciveMatrix;

  const T scaleX = static_cast<T>(1) / (tanHalfFovY * aspect);
  const T scaleY = static_cast<T>(1) / (tanHalfFovY);
  // clang-format off
  const T scaleZ          = -(farZ + nearZ) / (farZ - nearZ);                     // not the same (depends on handness + NO / LO)
  const T translateZ      = -(static_cast<T>(2) * farZ * nearZ) / (farZ - nearZ); // depends on NO / LO
  const T handednessScale = -static_cast<T>(1);                                   // depends on handness (-z)
  // clang-format on

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    perspeciveMatrix <<
      scaleX,  T(),     T(),         T(),
      T(),     scaleY,  T(),         T(),
      T(),     T(),     scaleZ,      handednessScale,
      T(),     T(),     translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    perspeciveMatrix <<
      scaleX,  T(),     T(),              T(),
      T(),     scaleY,  T(),              T(),
      T(),     T(),     scaleZ,           translateZ,
      T(),     T(),     handednessScale,  T();

    // clang-format on
  }

  return perspeciveMatrix;
}

/**
 * @brief Generates a right-handed perspective projection matrix with a depth
 * range of zero to one.
 *
 * Creates a 4x4 perspective projection matrix using the specified field of
 * view, aspect ratio, and near and far clipping planes.
 *
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param aspect Aspect ratio, defined as view space width divided by height.
 * @param nearZ Near clipping plane distance.
 * @param farZ Far clipping plane distance.
 * @return A 4x4 perspective projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhZo(T fovY, T aspect, T nearZ, T farZ)
    -> Matrix<T, 4, 4, Option> {
  // Validate aspect ratio to prevent division by zero
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspeciveMatrix;

  const T scaleX = static_cast<T>(1) / (tanHalfFovY * aspect);
  const T scaleY = static_cast<T>(1) / (tanHalfFovY);
  // clang-format off
  const T scaleZ          = -farZ / (farZ - nearZ);           // not the same (depends on handness + NO / LO)
  const T translateZ      = -(farZ * nearZ) / (farZ - nearZ); // depends on NO / LO
  const T handednessScale = -static_cast<T>(1);               // depends on handness (-z)
  // clang-format on

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    perspeciveMatrix <<
      scaleX,  T(),     T(),         T(),
      T(),     scaleY,  T(),         T(),
      T(),     T(),     scaleZ,      handednessScale,
      T(),     T(),     translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    perspeciveMatrix <<
      scaleX,  T(),     T(),              T(),
      T(),     scaleY,  T(),              T(),
      T(),     T(),     scaleZ,           translateZ,
      T(),     T(),     handednessScale,  T();
    // clang-format on
  }

  return perspeciveMatrix;
}

/**
 * @brief Generates a left-handed perspective projection matrix with a depth
 * range of negative one to one.
 *
 * Creates a 4x4 perspective projection matrix using the specified field of
 * view, aspect ratio, and near and far clipping planes.
 *
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param aspect Aspect ratio, defined as view space width divided by height.
 * @param nearZ Near clipping plane distance.
 * @param farZ Far clipping plane distance.
 * @return A 4x4 perspective projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhNo(T fovY, T aspect, T nearZ, T farZ)
    -> Matrix<T, 4, 4, Option> {
  // Validate aspect ratio to prevent division by zero
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspeciveMatrix;

  const T scaleX = static_cast<T>(1) / (tanHalfFovY * aspect);
  const T scaleY = static_cast<T>(1) / (tanHalfFovY);
  // clang-format off
  const T scaleZ          =  (farZ + nearZ) / (farZ - nearZ);                     // not the same (depends on handness + NO / LO)
  const T translateZ      = -(static_cast<T>(2) * farZ * nearZ) / (farZ - nearZ); // depends on NO / LO
  const T handednessScale =   static_cast<T>(1);                                  // depends on handness (z, not -z)
  // clang-format on

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    perspeciveMatrix <<
      scaleX,  T(),     T(),         T(),
      T(),     scaleY,  T(),         T(),
      T(),     T(),     scaleZ,      handednessScale,
      T(),     T(),     translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    perspeciveMatrix <<
      scaleX,  T(),     T(),              T(),
      T(),     scaleY,  T(),              T(),
      T(),     T(),     scaleZ,           translateZ,
      T(),     T(),     handednessScale,  T();
    // clang-format on
  }

  return perspeciveMatrix;
}

/**
 * @brief Generates a left-handed perspective projection matrix with a depth
 * range of zero to one.
 *
 * Creates a 4x4 perspective projection matrix using the specified field of
 * view, aspect ratio, and near and far clipping planes.
 *
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param aspect Aspect ratio, defined as view space width divided by height.
 * @param nearZ Near clipping plane distance.
 * @param farZ Far clipping plane distance.
 * @return A 4x4 perspective projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhZo(T fovY, T aspect, T nearZ, T farZ)
    -> Matrix<T, 4, 4, Option> {
  // Validate aspect ratio to prevent division by zero
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspeciveMatrix;

  const T scaleX = static_cast<T>(1) / (tanHalfFovY * aspect);
  const T scaleY = static_cast<T>(1) / (tanHalfFovY);
  // clang-format off
  const T scaleZ          =   farZ / (farZ - nearZ);          // not the same (depends on handness + NO / LO)
  const T translateZ      = -(farZ * nearZ) / (farZ - nearZ); // depends on NO / LO
  const T handednessScale =   static_cast<T>(1);              // depends on handness (z, not -z)
  // clang-format on

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    perspeciveMatrix <<
      scaleX,  T(),     T(),         T(),
      T(),     scaleY,  T(),         T(),
      T(),     T(),     scaleZ,      handednessScale,
      T(),     T(),     translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    perspeciveMatrix <<
      scaleX,  T(),     T(),              T(),
      T(),     scaleY,  T(),              T(),
      T(),     T(),     scaleZ,           translateZ,
      T(),     T(),     handednessScale,  T();
    // clang-format on
  }

  return perspeciveMatrix;
}

/**
 * @brief Generates a right-handed perspective projection matrix based on field
 * of view, width, and height with a depth range of negative one to one.
 *
 * Simplifies setting up the projection by directly specifying the viewport
 * dimensions and the near and far clipping planes.
 *
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param width Width of the viewport.
 * @param height Height of the viewport.
 * @param zNear Near clipping plane distance.
 * @param zFar Far clipping plane distance.
 * @return A 4x4 perspective projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhNo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveRhNo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * @brief Generates a right-handed perspective projection matrix based on field
 * of view, width, and height with a depth range of zero to one.
 *
 * Simplifies setting up the projection by directly specifying the viewport
 * dimensions and the near and far clipping planes.
 *
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param width Width of the viewport.
 * @param height Height of the viewport.
 * @param zNear Near clipping plane distance.
 * @param zFar Far clipping plane distance.
 * @return A 4x4 perspective projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhZo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveRhZo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * @brief Generates a left-handed perspective projection matrix based on field
 * of view, width, and height with a depth range of negative one to one.
 *
 * Simplifies setting up the projection by directly specifying the viewport
 * dimensions and the near and far clipping planes.
 *
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param width Width of the viewport.
 * @param height Height of the viewport.
 * @param zNear Near clipping plane distance.
 * @param zFar Far clipping plane distance.
 * @return A 4x4 perspective projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhNo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveLhNo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * @brief Generates a left-handed perspective projection matrix based on field
 * of view, width, and height with a depth range of zero to one.
 *
 * Simplifies setting up the projection by directly specifying the viewport
 * dimensions and the near and far clipping planes.
 *
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param width Width of the viewport.
 * @param height Height of the viewport.
 * @param zNear Near clipping plane distance.
 * @param zFar Far clipping plane distance.
 * @return A 4x4 perspective projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhZo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveLhZo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * @brief Generates a right-handed perspective projection matrix optimized for
 * rendering scenes with an infinite far plane.
 *
 * Creates a 4x4 perspective projection matrix using the specified field of
 * view, aspect ratio, and near clipping plane.
 *
 * @note RH-NO-Inf - Right-Handed, Negative One to One depth range, Infinite far
 * plane.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param aspect Aspect ratio, defined as view space width divided by height.
 * @param nearZ Near clipping plane distance.
 * @return A 4x4 perspective projection matrix optimized for an infinite far
 * plane.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhNoInf(T fovY, T aspect, T nearZ)
    -> Matrix<T, 4, 4, Option> {
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  // Explanation of matrix structure.
  // We use default perspective projection creation matrices, but for infinite
  // far plane we need to change matrix a little bit. As far approaches
  // infinity, (far / (far - near)) approaches 1, and (near / (far - near))
  // approaches 0. Thus:
  // 1) -(far + near) / (far - near) => -1
  // 2) -(2 * far * near) / (far - near) => -2 * near

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspectiveMatrix;

  const T scaleX = static_cast<T>(1) / (tanHalfFovY * aspect);
  const T scaleY = static_cast<T>(1) / tanHalfFovY;
  // clang-format off
  const T scaleZ          = -static_cast<T>(1);         // depends on handness (-z)
  const T translateZ      = -static_cast<T>(2) * nearZ; // depends on NO / LO  
  const T handednessScale = -static_cast<T>(1);         // depends on handness (-z)
  // clang-format on

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    perspectiveMatrix <<
      scaleX,  T(),     T(),         T(),
      T(),     scaleY,  T(),         T(),
      T(),     T(),     scaleZ,      handednessScale,
      T(),     T(),     translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    perspectiveMatrix <<
      scaleX,  T(),     T(),              T(),
      T(),     scaleY,  T(),              T(),
      T(),     T(),     scaleZ,           translateZ,
      T(),     T(),     handednessScale,  T();
    // clang-format on
  }
  return perspectiveMatrix;
}

/**
 * @brief Generates a right-handed perspective projection matrix optimized for
 * rendering scenes with an infinite far plane, using a depth range of zero to
 * one.
 *
 * Creates a 4x4 perspective projection matrix using the specified field of
 * view, aspect ratio, and near clipping plane. This matrix is optimized for
 * scenarios where the far clipping plane is at infinity, which can improve
 * depth precision and rendering of distant objects.
 *
 * @note RH-ZO-Inf - Right-Handed, Zero to One depth range, Infinite far plane.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param aspect Aspect ratio, defined as view space width divided by height.
 * @param nearZ Near clipping plane distance.
 * @return A 4x4 perspective projection matrix optimized for an infinite far
 * plane.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhZoInf(T fovY, T aspect, T nearZ)
    -> Matrix<T, 4, 4, Option> {
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  // Explanation of matrix structure.
  // We use default perspective projection creation matrices, but for infinite
  // far plane we need to change matrix a little bit. As far approaches
  // infinity, (far / (far - near)) approaches 1, and (near / (far - near))
  // approaches 0. Thus:
  // 1) -far / (far - near) => -1
  // 2) -(far * near) / (far - near) => -near

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspectiveMatrix;

  const T scaleX = static_cast<T>(1) / (tanHalfFovY * aspect);
  const T scaleY = static_cast<T>(1) / tanHalfFovY;
  // clang-format off
  const T scaleZ          = -static_cast<T>(1); // depends on handness (-z) 
  const T translateZ      = -nearZ;             // depends on NO / LO      
  const T handednessScale = -static_cast<T>(1); // depends on handness (-z)
  // clang-format on

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    perspectiveMatrix <<
      scaleX,  T(),     T(),         T(),
      T(),     scaleY,  T(),         T(),
      T(),     T(),     scaleZ,      handednessScale,
      T(),     T(),     translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    perspectiveMatrix <<
      scaleX,  T(),     T(),              T(),
      T(),     scaleY,  T(),              T(),
      T(),     T(),     scaleZ,           translateZ,
      T(),     T(),     handednessScale,  T();
    // clang-format on
  }
  return perspectiveMatrix;
}

/**
 * @brief Generates a left-handed perspective projection matrix optimized for
 * rendering scenes with an infinite far plane, using a depth range of negative
 * one to one.
 *
 * Creates a 4x4 perspective projection matrix using the specified field of
 * view, aspect ratio, and near clipping plane. This matrix is designed for
 * left-handed coordinate systems and is optimized for scenarios where the far
 * clipping plane is at infinity.
 *
 * @note LH-NO-Inf - Left-Handed, Negative One to One depth range, Infinite far
 * plane.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param aspect Aspect ratio, defined as view space width divided by height.
 * @param nearZ Near clipping plane distance.
 * @return A 4x4 perspective projection matrix optimized for an infinite far
 * plane.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhNoInf(T fovY, T aspect, T nearZ)
    -> Matrix<T, 4, 4, Option> {
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  // Explanation of matrix structure.
  // We use default perspective projection creation matrices, but for infinite
  // far plane we need to change matrix a little bit. As far approaches
  // infinity, (far / (far - near)) approaches 1, and (near / (far - near))
  // approaches 0. Thus:
  // 1) (far + near) / (far - near) => 1
  // 2) -(2 * far * near) / (far - near) => -2 * near

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspectiveMatrix;

  const T scaleX = static_cast<T>(1) / (tanHalfFovY * aspect);
  const T scaleY = static_cast<T>(1) / tanHalfFovY;
  // clang-format off
  const T scaleZ          =  static_cast<T>(1);         // depends on handness (z, not -z)
  const T translateZ      = -static_cast<T>(2) * nearZ; // depends on NO / LO              
  const T handednessScale =  static_cast<T>(1);         // depends on handness (z, not -z)
  // clang-format on

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    perspectiveMatrix <<
      scaleX,  T(),     T(),         T(),
      T(),     scaleY,  T(),         T(),
      T(),     T(),     scaleZ,      handednessScale,
      T(),     T(),     translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    perspectiveMatrix <<
      scaleX,  T(),     T(),              T(),
      T(),     scaleY,  T(),              T(),
      T(),     T(),     scaleZ,           translateZ,
      T(),     T(),     handednessScale,  T();
    // clang-format on
  }
  return perspectiveMatrix;
}

/**
 * @brief Produces a left-handed perspective projection matrix optimized for
 * scenes with an infinite far plane and a depth range of zero to one.
 *
 * This function creates a 4x4 perspective projection matrix using the specified
 * field of view, aspect ratio, and near clipping plane. It is tailored for
 * left-handed coordinate systems and assumes an infinite far plane, which is
 * useful for rendering scenes where the far distance is effectively limitless.
 *
 * @note LH-ZO-Inf - Left-Handed, Zero to One depth range, Infinite far plane.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param fovY Field of view in the y direction, in radians.
 * @param aspect Aspect ratio, defined as view space width divided by height.
 * @param nearZ Near clipping plane distance.
 * @return A 4x4 perspective projection matrix optimized for an infinite far
 * plane.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhZoInf(T fovY, T aspect, T nearZ)
    -> Matrix<T, 4, 4, Option> {
  assert(std::abs(aspect - std::numeric_limits<T>::epsilon())
         > static_cast<T>(0));

  // Explanation of matrix structure.
  // We use default perspective projection creation matrices, but for infinite
  // far plane we need to change matrix a little bit. As far approaches
  // infinity, (far / (far - near)) approaches 1, and (near / (far - near))
  // approaches 0. Thus:
  // 1) far / (far - near) => 1
  // 2) -(far * near) / (far - near) => -near

  const T tanHalfFovY = std::tan(fovY / static_cast<T>(2));

  Matrix<T, 4, 4, Option> perspectiveMatrix;

  const T scaleX = static_cast<T>(1) / (tanHalfFovY * aspect);
  const T scaleY = static_cast<T>(1) / tanHalfFovY;
  // clang-format off
  const T scaleZ          =  static_cast<T>(1); // depends on handness (z, not -z)
  const T translateZ      = -nearZ;             // depends on NO / LO              
  const T handednessScale =  static_cast<T>(1); // depends on handness (z, not -z)
  // clang-format on

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    perspectiveMatrix <<
      scaleX,  T(),     T(),         T(),
      T(),     scaleY,  T(),         T(),
      T(),     T(),     scaleZ,      handednessScale,
      T(),     T(),     translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    perspectiveMatrix <<
      scaleX,  T(),     T(),              T(),
      T(),     scaleY,  T(),              T(),
      T(),     T(),     scaleZ,           translateZ,
      T(),     T(),     handednessScale,  T();
    // clang-format on
  }

  return perspectiveMatrix;
}

// END: perspective projection creation matrix
// ----------------------------------------------------------------------------

/**
 * @brief Applies perspective division to a 4D vector for clip space and NDC
 * transformations.
 *
 * This function applies perspective division to a four-dimensional vector to
 * facilitate the transformation of points between clip space and normalized
 * device coordinates (NDC), essential for rendering 3D scenes with correct
 * perspective. The operation is crucial both for projecting points into NDC
 * during rendering and for reconstructing 3D positions from screen coordinates
 * through inverted transformations.
 *
 * @param vector The 4D vector to apply perspective division to.
 * @param tolerance The tolerance within which the w component is considered
 * effectively zero, preventing division in cases where it might lead to
 * numerical instability. This additional parameter is crucial for handling
 * floating-point inaccuracies, as dividing by values very close to zero can
 * result in Infinity or NaN values. The tolerance allows for a margin of error
 * in floating-point comparisons, ensuring that the division only occurs when
 * the w component is significantly different from zero.
 *
 * @tparam T A floating-point type representing the vector's element type.
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return The vector after perspective division.
 */
template <typename T, Options Option = Options::RowMajor>
  requires std::floating_point<T>
Vector<T, 4, Option> g_perspectiveDivide(const Vector<T, 4, Option>& vector,
                                         T tolerance = g_kDefaultTolerance) {
  if (static_cast<T>(1) != vector.w()
      && !g_isNearlyZero(vector.w(), tolerance)) {
    return vector / vector.w();
  }
  return vector;
}

// BEGIN: frustrum (perspective projection matrix that off center) creation functions
// ----------------------------------------------------------------------------------

/**
 * @brief Generates a right-handed frustum projection matrix with a depth range
 * of zero to one.
 *
 * Creates a 4x4 perspective projection matrix that defines a frustum (a
 * truncated pyramid) with the specified left, right, bottom, top, near, and far
 * clipping planes. This is useful for creating asymmetric perspective
 * projections, such as for stereo rendering or oblique projections.
 *
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements.
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param left Coordinate of the left vertical clipping plane.
 * @param right Coordinate of the right vertical clipping plane.
 * @param bottom Coordinate of the bottom horizontal clipping plane.
 * @param top Coordinate of the top horizontal clipping plane.
 * @param nearVal Distance to the near depth clipping plane (must be positive).
 * @param farVal Distance to the far depth clipping plane (must be positive).
 * @return A 4x4 frustum projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_frustumRhZo(T left, T right, T bottom, T top, T nearVal, T farVal)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> frustum;

  const T scaleX = (static_cast<T>(2) * nearVal) / (right - left);
  const T scaleY = (static_cast<T>(2) * nearVal) / (top - bottom);
  // depends on NO / ZO
  const T scaleZ = farVal / (nearVal - farVal);

  // depends on handness
  const T offsetX = (right + left) / (right - left);
  // depends on handness
  const T offsetY = (top + bottom) / (top - bottom);

  // depends on NO / ZO
  const T translateZ = -(farVal * nearVal) / (farVal - nearVal);

  // depends on handness
  const T handedness = -static_cast<T>(1);

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    frustum <<
      scaleX,   T(),      T(),         T(),
      T(),      scaleY,   T(),         T(),
      offsetX,  offsetY,  scaleZ,      handedness,
      T(),      T(),      translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    frustum <<
      scaleX,  T(),     offsetX,     T(),
      T(),     scaleY,  offsetY,     T(),
      T(),     T(),     scaleZ,      translateZ,
      T(),     T(),     handedness,  T();
    // clang-format on
  }

  return frustum;
}

/**
 * @brief Generates a right-handed frustum projection matrix with a depth range
 * of negative one to one.
 *
 * Creates a 4x4 perspective projection matrix that defines a frustum (a
 * truncated pyramid) using the specified left, right, bottom, top, near, and
 * far clipping planes. This function is useful for creating asymmetric
 * perspective projections, such as for stereo rendering or oblique projections,
 * in a right-handed coordinate system with a depth range from -1 to 1.
 *
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param left The coordinate for the left vertical clipping plane.
 * @param right The coordinate for the right vertical clipping plane.
 * @param bottom The coordinate for the bottom horizontal clipping plane.
 * @param top The coordinate for the top horizontal clipping plane.
 * @param nearVal The distance to the near depth clipping plane (must be
 * positive).
 * @param farVal The distance to the far depth clipping plane (must be
 * positive).
 * @return A 4x4 frustum projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_frustumRhNo(T left, T right, T bottom, T top, T nearVal, T farVal)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> frustum;

  const T scaleX = (static_cast<T>(2) * nearVal) / (right - left);
  const T scaleY = (static_cast<T>(2) * nearVal) / (top - bottom);
  // depends on NO / ZO
  const T scaleZ = -(farVal + nearVal) / (farVal - nearVal);

  // depends on handness
  const T offsetX = (right + left) / (right - left);
  // depends on handness
  const T offsetY = (top + bottom) / (top - bottom);

  // depends on NO / ZO
  const T translateZ
      = -(static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);

  // depends on handness
  const T handedness = -static_cast<T>(1);

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    frustum <<
      scaleX,   T(),      T(),         T(),
      T(),      scaleY,   T(),         T(),
      offsetX,  offsetY,  scaleZ,      handedness,
      T(),      T(),      translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    frustum <<
      scaleX,  T(),     offsetX,     T(),
      T(),     scaleY,  offsetY,     T(),
      T(),     T(),     scaleZ,      translateZ,
      T(),     T(),     handedness,  T();
    // clang-format on
  }
  return frustum;
}

/**
 * @brief Generates a left-handed frustum projection matrix with a depth range
 * of zero to one.
 *
 * Creates a 4x4 perspective projection matrix that defines a frustum (a
 * truncated pyramid) using the specified left, right, bottom, top, near, and
 * far clipping planes. This is useful for creating asymmetric perspective
 * projections, such as for shadow mapping or oblique projections, in a
 * left-handed coordinate system with a depth range from 0 to 1.
 *
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param left The coordinate for the left vertical clipping plane.
 * @param right The coordinate for the right vertical clipping plane.
 * @param bottom The coordinate for the bottom horizontal clipping plane.
 * @param top The coordinate for the top horizontal clipping plane.
 * @param nearVal The distance to the near depth clipping plane (must be
 * positive).
 * @param farVal The distance to the far depth clipping plane (must be
 * positive).
 * @return A 4x4 frustum projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_frustumLhZo(T left, T right, T bottom, T top, T nearVal, T farVal)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> frustum;

  const T scaleX = (static_cast<T>(2) * nearVal) / (right - left);
  const T scaleY = (static_cast<T>(2) * nearVal) / (top - bottom);
  // depends on NO / ZO + handness
  const T scaleZ = farVal / (farVal - nearVal);

  // depends on handness
  const T offsetX = -(right + left) / (right - left);
  // depends on handness
  const T offsetY = -(top + bottom) / (top - bottom);

  // depends on NO / ZO
  const T translateZ = -(farVal * nearVal) / (farVal - nearVal);

  // depends on handness
  const T handedness = static_cast<T>(1);

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    frustum <<
      scaleX,   T(),      T(),         T(),
      T(),      scaleY,   T(),         T(),
      offsetX,  offsetY,  scaleZ,      handedness,
      T(),      T(),      translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    frustum <<
      scaleX,  T(),     offsetX,     T(),
      T(),     scaleY,  offsetY,     T(),
      T(),     T(),     scaleZ,      translateZ,
      T(),     T(),     handedness,  T();
    // clang-format on
  }
  return frustum;
}

/**
 * @brief Generates a left-handed frustum projection matrix with a depth range
 * of negative one to one.
 *
 * Creates a 4x4 perspective projection matrix that defines a frustum (a
 * truncated pyramid) using the specified left, right, bottom, top, near, and
 * far clipping planes. This function is useful for creating asymmetric
 * perspective projections, such as for shadow mapping or oblique projections,
 * in a left-handed coordinate system with a depth range from -1 to 1.
 *
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param left The coordinate for the left vertical clipping plane.
 * @param right The coordinate for the right vertical clipping plane.
 * @param bottom The coordinate for the bottom horizontal clipping plane.
 * @param top The coordinate for the top horizontal clipping plane.
 * @param nearVal The distance to the near depth clipping plane (must be
 * positive).
 * @param farVal The distance to the far depth clipping plane (must be
 * positive).
 * @return A 4x4 frustum projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_frustumLhNo(T left, T right, T bottom, T top, T nearVal, T farVal)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> frustum;

  const T scaleX = (static_cast<T>(2) * nearVal) / (right - left);
  const T scaleY = (static_cast<T>(2) * nearVal) / (top - bottom);
  // depends on NO / ZO
  const T scaleZ = (farVal + nearVal) / (farVal - nearVal);

  // depends on handness
  const T offsetX = -(right + left) / (right - left);
  // depends on handness
  const T offsetY = -(top + bottom) / (top - bottom);

  // depends on NO / ZO
  const T translateZ
      = -(static_cast<T>(2) * farVal * nearVal) / (farVal - nearVal);

  // depends on handness
  const T handedness = static_cast<T>(1);

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    frustum <<
      scaleX,   T(),      T(),         T(),
      T(),      scaleY,   T(),         T(),
      offsetX,  offsetY,  scaleZ,      handedness,
      T(),      T(),      translateZ,  T();
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    frustum <<
      scaleX,  T(),     offsetX,     T(),
      T(),     scaleY,  offsetY,     T(),
      T(),     T(),     scaleZ,      translateZ,
      T(),     T(),     handedness,  T();
    // clang-format on
  }

  return frustum;
}

// END: frustrum (perspective projection matrix that off center) creation functions
// --------------------------------------------------------------------------------

// BEGIN: orthographic projection creation matrix
// ----------------------------------------------------------------------------

// TODO: add ortho functions (LH/RH) that takes (left, right, bottom, top) w/o
// near / far

/**
 * @brief Generates a left-handed orthographic projection matrix with a depth
 * range of zero to one.
 *
 * Creates a 4x4 orthographic projection matrix using the specified left, right,
 * bottom, top, near, and far clipping planes. This projection maintains
 * parallel lines and is commonly used for UI rendering, CAD applications, and
 * shadow maps in a left-handed coordinate system with a depth range from 0
 * to 1.
 *
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param left The coordinate for the left vertical clipping plane.
 * @param right The coordinate for the right vertical clipping plane.
 * @param bottom The coordinate for the bottom horizontal clipping plane.
 * @param top The coordinate for the top horizontal clipping plane.
 * @param nearZ The distance to the near depth clipping plane.
 * @param farZ The distance to the far depth clipping plane.
 * @return A 4x4 orthographic projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoLhZo(T left, T right, T bottom, T top, T nearZ, T farZ)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> orthographicMat;

  const T scaleX = static_cast<T>(2) / (right - left);
  const T scaleY = static_cast<T>(2) / (top - bottom);
  // depends on handness + ZO / NO
  const T scaleZ = static_cast<T>(1) / (farZ - nearZ);

  const T translateX = -(right + left) / (right - left);
  const T translateY = -(top + bottom) / (top - bottom);
  // depends on ZO / NO
  const T translateZ = -nearZ / (farZ - nearZ);

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    orthographicMat <<
      scaleX,      T(),         T(),         T(),
      T(),         scaleY,      T(),         T(),
      T(),         T(),         scaleZ,      T(),
      translateX,  translateY,  translateZ,  T(1);
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    orthographicMat <<
      scaleX,  T(),     T(),     translateX,
      T(),     scaleY,  T(),     translateY,
      T(),     T(),     scaleZ,  translateZ,
      T(),     T(),     T(),     T(1);
    // clang-format on
  }

  return orthographicMat;
}

/**
 * @brief Generates a left-handed orthographic projection matrix with a depth
 * range of negative one to one.
 *
 * Creates a 4x4 orthographic projection matrix using the specified left, right,
 * bottom, top, near, and far clipping planes. This projection maintains
 * parallel lines and is suitable for UI rendering, CAD applications, and 2D
 * games in a left-handed coordinate system with a depth range from -1 to 1.
 *
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param left The coordinate for the left vertical clipping plane.
 * @param right The coordinate for the right vertical clipping plane.
 * @param bottom The coordinate for the bottom horizontal clipping plane.
 * @param top The coordinate for the top horizontal clipping plane.
 * @param nearZ The distance to the near depth clipping plane.
 * @param farZ The distance to the far depth clipping plane.
 * @return A 4x4 orthographic projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoLhNo(T left, T right, T bottom, T top, T nearZ, T farZ)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> orthographicMat;

  const T scaleX = static_cast<T>(2) / (right - left);
  const T scaleY = static_cast<T>(2) / (top - bottom);
  // depends on handness + ZO / NO
  const T scaleZ = static_cast<T>(2) / (farZ - nearZ);

  const T translateX = -(right + left) / (right - left);
  const T translateY = -(top + bottom) / (top - bottom);
  // depends on ZO / NO
  const T translateZ = -(farZ + nearZ) / (farZ - nearZ);

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    orthographicMat <<
      scaleX,      T(),         T(),         T(),
      T(),         scaleY,      T(),         T(),
      T(),         T(),         scaleZ,      T(),
      translateX,  translateY,  translateZ,  T(1);
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    orthographicMat <<
      scaleX,  T(),     T(),     translateX,
      T(),     scaleY,  T(),     translateY,
      T(),     T(),     scaleZ,  translateZ,
      T(),     T(),     T(),     T(1);
    // clang-format on
  }

  return orthographicMat;
}

/**
 * @brief Generates a right-handed orthographic projection matrix with a depth
 * range of zero to one.
 *
 * Creates a 4x4 orthographic projection matrix using the specified left, right,
 * bottom, top, near, and far clipping planes. This projection is useful for
 * rendering UI elements, 2D games, and CAD models in a right-handed coordinate
 * system with a depth range from 0 to 1.
 *
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param left The coordinate for the left vertical clipping plane.
 * @param right The coordinate for the right vertical clipping plane.
 * @param bottom The coordinate for the bottom horizontal clipping plane.
 * @param top The coordinate for the top horizontal clipping plane.
 * @param nearZ The distance to the near depth clipping plane.
 * @param farZ The distance to the far depth clipping plane.
 * @return A 4x4 orthographic projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoRhZo(T left, T right, T bottom, T top, T nearZ, T farZ)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> orthographicMat;

  const T scaleX = static_cast<T>(2) / (right - left);
  const T scaleY = static_cast<T>(2) / (top - bottom);
  // depends on handness + ZO / NO
  const T scaleZ = -static_cast<T>(1) / (farZ - nearZ);

  const T translateX = -(right + left) / (right - left);
  const T translateY = -(top + bottom) / (top - bottom);
  // depends on ZO / NO
  const T translateZ = -nearZ / (farZ - nearZ);

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    orthographicMat <<
      scaleX,      T(),         T(),         T(),
      T(),         scaleY,      T(),         T(),
      T(),         T(),         scaleZ,      T(),
      translateX,  translateY,  translateZ,  T(1);
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    orthographicMat <<
      scaleX,  T(),     T(),     translateX,
      T(),     scaleY,  T(),     translateY,
      T(),     T(),     scaleZ,  translateZ,
      T(),     T(),     T(),     T(1);
    // clang-format on
  }

  return orthographicMat;
}

/**
 * @brief Generates a right-handed orthographic projection matrix with a depth
 * range of negative one to one.
 *
 * Creates a 4x4 orthographic projection matrix using the specified left, right,
 * bottom, top, near, and far clipping planes. This projection maintains
 * parallel lines and is commonly used for rendering UI elements, 2D games, and
 * CAD models in a right-handed coordinate system with a depth range from -1
 * to 1.
 *
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float`, `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param left The coordinate for the left vertical clipping plane.
 * @param right The coordinate for the right vertical clipping plane.
 * @param bottom The coordinate for the bottom horizontal clipping plane.
 * @param top The coordinate for the top horizontal clipping plane.
 * @param nearZ The distance to the near depth clipping plane.
 * @param farZ The distance to the far depth clipping plane.
 * @return A 4x4 orthographic projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoRhNo(T left, T right, T bottom, T top, T nearZ, T farZ)
    -> Matrix<T, 4, 4, Option> {
  Matrix<T, 4, 4, Option> orthographicMat;

  const T scaleX = static_cast<T>(2) / (right - left);
  const T scaleY = static_cast<T>(2) / (top - bottom);
  // depends on handness + ZO / NO
  const T scaleZ = -static_cast<T>(2) / (farZ - nearZ);

  const T translateX = -(right + left) / (right - left);
  const T translateY = -(top + bottom) / (top - bottom);
  // depends on ZO / NO
  const T translateZ = -(farZ + nearZ) / (farZ - nearZ);

  if constexpr (Option == Options::RowMajor) {
    // clang-format off
    orthographicMat <<
      scaleX,      T(),         T(),         T(),
      T(),         scaleY,      T(),         T(),
      T(),         T(),         scaleZ,      T(),
      translateX,  translateY,  translateZ,  T(1);
    // clang-format on
  } else if constexpr (Option == Options::ColumnMajor) {
    // clang-format off
    orthographicMat <<
      scaleX,  T(),     T(),     translateX,
      T(),     scaleY,  T(),     translateY,
      T(),     T(),     scaleZ,  translateZ,
      T(),     T(),     T(),     T(1);
    // clang-format on
  }

  return orthographicMat;
}

// TODO: remove code duplication from the functions below

/**
 * @brief Generates a left-handed orthographic projection matrix based on the
 * given width and height, with a depth range of zero to one.
 *
 * This function simplifies the creation of an orthographic projection matrix by
 * directly specifying the viewport dimensions (width and height) and the near
 * and far clipping planes. It calculates the left, right, bottom, and top
 * parameters internally and then calls the standard `g_orthoLhZo` function to
 * generate the matrix. This projection is commonly used in 2D rendering and
 * UI, particularly in graphics APIs that utilize a 0 to 1 depth range, such as
 * DirectX.
 *
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float` or `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param width The width of the viewport.
 * @param height The height of the viewport.
 * @param zNear The distance to the near clipping plane.
 * @param zFar The distance to the far clipping plane.
 * @return A 4x4 left-handed orthographic projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoLhZo(T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  T halfWidth  = width / static_cast<T>(2);
  T halfHeight = height / static_cast<T>(2);

  T left   = -halfWidth;
  T right  = halfWidth;
  T bottom = -halfHeight;
  T top    = halfHeight;

  return g_orthoLhZo<T, Option>(left, right, bottom, top, zNear, zFar);
}

/**
 * @brief Generates a left-handed orthographic projection matrix based on the
 * given width and height, with a depth range of negative one to one.
 *
 * This function simplifies the creation of an orthographic projection matrix by
 * directly specifying the viewport dimensions (width and height) and the near
 * and far clipping planes. It calculates the left, right, bottom, and top
 * parameters internally and then calls the standard `g_orthoLhNo` function to
 * generate the matrix. This is useful for setting up 2D rendering or UI
 * elements in a 3D space.
 *
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float` or `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param width The width of the viewport.
 * @param height The height of the viewport.
 * @param zNear The distance to the near clipping plane.
 * @param zFar The distance to the far clipping plane.
 * @return A 4x4 left-handed orthographic projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoLhNo(T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  T halfWidth  = width / static_cast<T>(2);
  T halfHeight = height / static_cast<T>(2);

  T left   = -halfWidth;
  T right  = halfWidth;
  T bottom = -halfHeight;
  T top    = halfHeight;

  return g_orthoLhNo<T, Option>(left, right, bottom, top, zNear, zFar);
}

/**
 * @brief Generates a right-handed orthographic projection matrix based on the
 * given width and height, with a depth range of zero to one.
 *
 * This function simplifies the creation of an orthographic projection matrix by
 * directly specifying the viewport dimensions (width and height) and the near
 * and far clipping planes. It internally calculates the left, right, bottom,
 * and top parameters and then calls the standard `g_orthoRhZo` function to
 * generate the matrix. This is particularly useful for setting up orthographic
 * projections in graphics APIs that use a depth range from 0 to 1, such as
 * DirectX.
 *
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float` or `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param width The width of the viewport.
 * @param height The height of the viewport.
 * @param zNear The distance to the near clipping plane.
 * @param zFar The distance to the far clipping plane.
 * @return A 4x4 right-handed orthographic projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoRhZo(T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  T halfWidth  = width / static_cast<T>(2);
  T halfHeight = height / static_cast<T>(2);

  T left   = -halfWidth;
  T right  = halfWidth;
  T bottom = -halfHeight;
  T top    = halfHeight;

  return g_orthoRhZo<T, Option>(left, right, bottom, top, zNear, zFar);
}

/**
 * @brief Generates a right-handed orthographic projection matrix based on the
 * given width and height, with a depth range of negative one to one.
 *
 * This function simplifies the setup of an orthographic projection by allowing
 * you to specify the viewport dimensions directly. It calculates the necessary
 * parameters internally and then calls the standard `g_orthoRhNo` function to
 * create the projection matrix. This is useful for rendering 2D elements in a
 * 3D space or for applications like CAD programs.
 *
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 *
 * @tparam T The data type of the matrix elements (e.g., `float` or `double`).
 * @tparam Option Specifies whether the matrix is row-major or column-major.
 * @param width The width of the viewport.
 * @param height The height of the viewport.
 * @param zNear The distance to the near clipping plane.
 * @param zFar The distance to the far clipping plane.
 * @return A 4x4 right-handed orthographic projection matrix.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_orthoRhNo(T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  T halfWidth  = width / static_cast<T>(2);
  T halfHeight = height / static_cast<T>(2);

  T left   = -halfWidth;
  T right  = halfWidth;
  T bottom = -halfHeight;
  T top    = halfHeight;

  return g_orthoRhNo<T, Option>(left, right, bottom, top, zNear, zFar);
}

// END: orthographic projection creation matrix
// ----------------------------------------------------------------------------

/**
 * @brief Transforms a 3D point using a specified transformation matrix and
 * applies perspective division.
 *
 * This function multiplies a 3D point by a 4x4 transformation matrix. It first
 * converts the 3D point into a 4D homogeneous coordinate by setting the
 * w-component to 1. After the transformation, it performs a perspective
 * division to convert back to 3D coordinates, which is essential when using
 * perspective projection matrices.
 *
 * @note This function automatically applies perspective division for points
 * transformed with a perspective projection matrix. The default tolerance is
 * used for `g_perspectiveDivide`. Consider adding a parameter to adjust this if
 * needed.
 *
 * @tparam T The floating-point data type of the point and matrix elements
 * (e.g., `float` or `double`).
 * @tparam Option Specifies whether the point and matrix are row-major or
 * column-major.
 * @param point The 3D point to be transformed.
 * @param matrix The 4x4 transformation matrix.
 * @return The transformed 3D point after applying the transformation matrix and
 * perspective division.
 */
template <typename T, Options Option = Options::RowMajor>
  requires std::floating_point<T>
Point<T, 3, Option> g_transformPoint(const Point<T, 3, Option>&     point,
                                     const Matrix<T, 4, 4, Option>& matrix) {
  // TODO: currently in this implementation default tolerance used for
  // g_perspectiveDivide. Consider add pararmeter if there will be a need
  Point<T, 4, Option> result  = Point<T, 4, Option>(point, 1);
  result                     *= matrix;
  // applied when perspective projection matrix is used
  result = g_perspectiveDivide(result);
  return result.template resizedCopy<3>();
}

/**
 * @brief Transforms a 3D vector using a specified transformation matrix.
 *
 * This function multiplies a 3D vector by a 4x4 transformation matrix. It first
 * converts the 3D vector into a 4D homogeneous coordinate by setting the
 * w-component to 0. This ensures that the vector is only affected by the linear
 * transformations (rotation and scaling) and not by translations in the matrix.
 *
 * @tparam T The data type of the vector and matrix elements (e.g., `float` or
 * `double`).
 * @tparam Option Specifies whether the vector and matrix are row-major or
 * column-major.
 * @param vector The 3D vector to be transformed.
 * @param matrix The 4x4 transformation matrix.
 * @return The transformed 3D vector after applying the transformation matrix.
 */
template <typename T, Options Option = Options::RowMajor>
Vector<T, 3, Option> g_transformVector(const Vector<T, 3, Option>&    vector,
                                       const Matrix<T, 4, 4, Option>& matrix) {
  auto result  = Vector<T, 4, Option>(vector, 0);
  result      *= matrix;
  return result.template resizedCopy<3>();
}

// BEGIN: global util vector objects
// ----------------------------------------------------------------------------

// TODO: consider moving directional vector variables to a separate file

// TODO: make these constexpr

/**
 * @brief Returns a unit vector pointing upwards along the positive Y-axis.
 *
 * Generates a vector with all components zero except for the Y-component, which
 * is set to one. This function requires the vector size to be at least 2.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector (must be at least 2).
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A vector pointing upwards along the Y-axis.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_upVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.y() = 1;
  return vec;
}

/**
 * @brief Returns a unit vector pointing downwards along the negative Y-axis.
 *
 * Generates a vector with all components zero except for the Y-component, which
 * is set to negative one. This function requires the vector size to be at
 * least 2.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector (must be at least 2).
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A vector pointing downwards along the Y-axis.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_downVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.y() = -1;
  return vec;
}


/**
 * @brief Returns a unit vector pointing to the right along the positive X-axis.
 *
 * Generates a vector with all components zero except for the X-component, which
 * is set to one. This function requires the vector size to be at least 2.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector (must be at least 2).
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A vector pointing to the right along the X-axis.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_rightVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.x() = 1;
  return vec;
}

/**
 * @brief Returns a unit vector pointing to the left along the negative X-axis.
 *
 * Generates a vector with all components zero except for the X-component, which
 * is set to negative one. This function requires the vector size to be at
 * least 2.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector (must be at least 2).
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A vector pointing to the left along the X-axis.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_leftVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.x() = -1;
  return vec;
}

/**
 * @brief Returns a unit vector pointing forward along the positive Z-axis.
 *
 * Generates a vector with all components zero except for the Z-component, which
 * is set to one. This function requires the vector size to be at least 3.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector (must be at least 3).
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A vector pointing forward along the Z-axis.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 3>
auto g_forwardVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.z() = 1;
  return vec;
}

/**
 * @brief Returns a unit vector pointing backward along the negative Z-axis.
 *
 * Generates a vector with all components zero except for the Z-component, which
 * is set to negative one. This function requires the vector size to be at
 * least 3.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector (must be at least 3).
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A vector pointing backward along the Z-axis.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 3>
auto g_backwardVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.z() = -1;
  return vec;
}

/**
 * @brief Returns a zero vector of the specified size.
 *
 * Generates a vector where all components are set to zero.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector.
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A zero vector.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
auto g_zeroVector() -> const Vector<T, Size, Option> {
  return Vector<T, Size, Option>(0);
}

/**
 * @brief Returns a vector with all components set to one.
 *
 * Generates a vector where all components are set to one.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector.
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A vector with all components equal to one.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
auto g_oneVector() -> const Vector<T, Size, Option> {
  return Vector<T, Size, Option>(1);
}

/**
 * @brief Returns a unit vector with all components equal, pointing diagonally.
 *
 * Generates a vector where all components are equal and the vector is
 * normalized to have a magnitude of one.
 *
 * @tparam T The data type of the vector elements.
 * @tparam Size The size of the vector.
 * @tparam Option Specifies whether the vector is row-major or column-major.
 * @return A unit vector with equal components.
 */
template <typename T, std::size_t Size, Options Option = Options::RowMajor>
auto g_unitVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(1);
  return vec.normalized();
}

// END: global util vector objects
// ----------------------------------------------------------------------------

// BEGIN: intersections functionality (currently in this file)
// ----------------------------------------------------------------------------

/**
 * @brief Represents a ray in 3D space for intersection calculations
 *
 * A ray is defined by an origin point and a normalized direction vector.
 *
 * @tparam T The data type of the ray components
 * @tparam Option The memory layout option
 */
template <typename T, Options Option = Options::RowMajor>
class Ray {
  public:
  Ray() = default;

  /**
   * @brief Constructs a ray from origin and direction
   *
   * @param origin The starting point of the ray
   * @param direction The direction vector (will be normalized)
   */
  Ray(const Point3<T, Option>& origin, const Vector3<T, Option>& direction)
      : m_origin(origin)
      , m_direction(direction.normalized()) {}

  [[nodiscard]] auto origin() const -> const Point3<T, Option>& {
    return m_origin;
  }

  [[nodiscard]] auto direction() const -> const Vector3<T, Option>& {
    return m_direction;
  }

  /**
   * @brief Calculates a point along the ray at distance t
   *
   * @param t The distance along the ray
   * @return The point at origin + t * direction
   */
  [[nodiscard]] auto pointAt(T t) const -> Point3<T, Option> {
    return m_origin + m_direction * t;
  }

  private:
  Point3<T, Option>  m_origin;
  Vector3<T, Option> m_direction;
};

// Type aliases
template <Options Option = Options::RowMajor>
using Rayf = Ray<float, Option>;

template <Options Option = Options::RowMajor>
using Rayd = Ray<double, Option>;

/**
 * @brief Simple intersection result structure
 *
 * Contains information about ray-primitive intersection
 */
template <typename T>
struct IntersectionResult {
  bool       hit      = false;
  T          distance = 0;
  Point3<T> point;

  explicit operator bool() const { return hit; }
};

/**
 * @brief Tests ray-AABB (Axis-Aligned Bounding Box) intersection
 *
 * Uses the slab method for efficient AABB intersection testing.
 * This is typically used as a broad-phase collision detection.
 *
 * @param ray The ray to test
 * @param min Minimum corner of the box
 * @param max Maximum corner of the box
 * @return Intersection result with hit information
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rayAABBintersect(const Ray<T, Option>&     ray,
                        const Point3<T, Option>& min,
                        const Point3<T, Option>& max)
    -> IntersectionResult<T> {
  IntersectionResult<T> result;

  T tmin = 0;
  T tmax = std::numeric_limits<T>::max();

  for (int i = 0; i < 3; ++i) {
    if (g_abs(ray.direction()(i)) < std::numeric_limits<T>::epsilon()) {
      // Ray parallel to slab
      if (ray.origin()(i) < min(i) || ray.origin()(i) > max(i)) {
        return result;
      }
    } else {
      T invD = T(1) / ray.direction()(i);
      T t0   = (min(i) - ray.origin()(i)) * invD;
      T t1   = (max(i) - ray.origin()(i)) * invD;

      if (invD < 0) {
        std::swap(t0, t1);
      }

      tmin = std::max(tmin, t0);
      tmax = std::min(tmax, t1);

      if (tmax <= tmin) {
        return result;
      }
    }
  }

  result.hit      = true;
  result.distance = tmin;
  result.point    = ray.pointAt(tmin);
  return result;
}

/**
 * @brief Tests ray-triangle intersection using Moller-Trumbore algorithm
 *
 * This is used for precise mesh intersection testing.
 * Typically used as narrow-phase collision detection after AABB test.
 *
 * @param ray The ray to test
 * @param v0 First vertex of triangle
 * @param v1 Second vertex of triangle
 * @param v2 Third vertex of triangle
 * @return Intersection result with hit information
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rayTriangleintersect(const Ray<T, Option>&     ray,
                            const Point3<T, Option>& v0,
                            const Point3<T, Option>& v1,
                            const Point3<T, Option>& v2)
    -> IntersectionResult<T> {
  IntersectionResult<T> result;

  constexpr T kEpsilon = std::numeric_limits<T>::epsilon();

  Vector3<T, Option> edge1 = v1 - v0;
  Vector3<T, Option> edge2 = v2 - v0;
  Vector3<T, Option> h     = ray.direction().cross(edge2);
  T                   a     = edge1.dot(h);

  if (a > -kEpsilon && a < kEpsilon) {
    return result;  // Ray is parallel to triangle
  }

  T                   f = T(1) / a;
  Vector3<T, Option> s = ray.origin() - v0;
  T                   u = f * s.dot(h);

  if (u < 0 || u > 1) {
    return result;
  }

  Vector3<T, Option> q = s.cross(edge1);
  T                   v = f * ray.direction().dot(q);

  if (v < 0 || u + v > 1) {
    return result;
  }

  T t = f * edge2.dot(q);

  if (t > kEpsilon) {
    result.hit      = true;
    result.distance = t;
    result.point    = ray.pointAt(t);
    return result;
  }

  return result;
}

/**
 * @brief Tests ray-sphere intersection
 *
 * Useful for bounding sphere tests or spherical objects.
 *
 * @param ray The ray to test
 * @param center Sphere center
 * @param radius Sphere radius
 * @return Intersection result with hit information
 */
template <typename T, Options Option = Options::RowMajor>
auto g_raySphereintersect(const Ray<T, Option>&     ray,
                          const Point3<T, Option>& center,
                          T radius) -> IntersectionResult<T> {
  IntersectionResult<T> result;

  Vector3<T, Option> oc = ray.origin() - center;
  T                   a  = ray.direction().dot(ray.direction());
  T                   b  = T(2) * oc.dot(ray.direction());
  T                   c  = oc.dot(oc) - radius * radius;

  T discriminant = b * b - T(4) * a * c;

  if (discriminant < 0) {
    return result;
  }

  T sqrtDiscriminant = std::sqrt(discriminant);
  T t1               = (-b - sqrtDiscriminant) / (T(2) * a);
  T t2               = (-b + sqrtDiscriminant) / (T(2) * a);

  // Take the closest positive intersection
  T t = (t1 > 0) ? t1 : ((t2 > 0) ? t2 : T(-1));

  if (t > 0) {
    result.hit      = true;
    result.distance = t;
    result.point    = ray.pointAt(t);
  }

  return result;
}

/**
 * @brief Creates a ray from screen coordinates
 *
 * Converts screen/mouse coordinates to a ray in world space.
 * Essential for mouse picking functionality.
 *
 * @param x Screen x coordinate (0 to width)
 * @param y Screen y coordinate (0 to height)
 * @param width Screen width
 * @param height Screen height
 * @param viewMatrix View matrix
 * @param projectionMatrix Projection matrix
 * @return Ray in world space
 */
template <typename T, Options Option = Options::RowMajor>
auto g_screenToRay(T                              x,
                   T                              y,
                   T                              width,
                   T                              height,
                   const Matrix<T, 4, 4, Option>& viewMatrix,
                   const Matrix<T, 4, 4, Option>& projectionMatrix)
    -> Ray<T, Option> {
  // Convert to normalized device coordinates (-1 to 1)
  T ndcX = (T(2) * x) / width - T(1);
  T ndcY = T(1) - (T(2) * y) / height;  // Flip Y

  // Create points in clip space
  Vector4<T, Option> nearPoint(ndcX, ndcY, T(-1), T(1));
  Vector4<T, Option> farPoint(ndcX, ndcY, T(1), T(1));

  // Transform to world space
  auto invProjView = (projectionMatrix * viewMatrix).inverse();

  Vector4<T, Option> worldNear  = nearPoint;
  worldNear                     *= invProjView;
  worldNear                      = g_perspectiveDivide(worldNear);

  Vector4<T, Option> worldFar  = farPoint;
  worldFar                     *= invProjView;
  worldFar                      = g_perspectiveDivide(worldFar);

  // Create ray
  Point3<T, Option>  origin    = worldNear.template resizedCopy<3>();
  Vector3<T, Option> direction = (worldFar.template resizedCopy<3>()
                                   - worldNear.template resizedCopy<3>())
                                      .normalized();

  return Ray<T, Option>(origin, direction);
}

// END: intersections functionality
// ----------------------------------------------------------------------------

}  // namespace math

#endif