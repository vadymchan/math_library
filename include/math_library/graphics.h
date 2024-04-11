/**
 * @file graphics.h
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

template <typename T, Options Option = Options::RowMajor>
auto g_translate(const Vector<T, 3, Option>& translation)
    -> Matrix<T, 4, 4, Option> {
  return g_translate<T, Option>(
      translation.x(), translation.y(), translation.z());
}

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

template <typename T, Options Option>
void g_addTranslate(Matrix<T, 4, 4, Option>&    matrix,
                    const Vector<T, 3, Option>& translation) {
  g_addTranslate(matrix, translation.x(), translation.y(), translation.z());
}

template <typename T, Options Option>
void g_setTranslate(Matrix<T, 4, 4, Option>& matrix, T dx, T dy, T dz) {
  Vector<T, 4, Option> translation(dx, dy, dz, matrix(3, 3));
  if constexpr (Option == Options::RowMajor) {
    matrix.setRow<3>(translation);
  } else if constexpr (Option == Options::ColumnMajor) {
    matrix.setColumn<3>(translation);
  }
}

template <typename T, Options Option>
void g_setTranslate(Matrix<T, 4, 4, Option>&    matrix,
                    const Vector<T, 3, Option>& translation) {
  g_setTranslate(matrix, translation.x(), translation.y(), translation.z());
}

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

template <typename T, Options Option = Options::RowMajor>
auto g_scale(const Vector<T, 3, Option>& scale) -> Matrix<T, 4, 4, Option> {
  return g_scale<T, Option>(scale.x(), scale.y(), scale.z());
}

// BEGIN: rotation matrix creation functions
// ----------------------------------------------------------------------------
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
 * @param axis The 3D vector representing the axis of rotation.
 * @param angle The rotation angle around the axis, in radians.
 *
 * @note This function is designed for right-handed coordinate systems. It
 * automatically normalizes the axis of rotation.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_rotateRh(const Vector<T, 3, Option>& axis, T angle)
    -> Matrix<T, 4, 4, Option> {
  const T kCosAngle    = std::cos(angle);
  const T kSinAngle    = std::sin(angle);
  const T kOneMinusCos = 1 - kCosAngle;

#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto normalizedAxis = axis;
  normalizedAxis.normalize();
#else
  auto normalizedAxis = axis.normalize();
#endif
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

template <typename T, Options Option = Options::RowMajor>
auto g_rotateLhX(T angle) -> Matrix<T, 4, 4, Option> {
  return g_rotateRhX<T, Option>(-angle);
}

template <typename T, Options Option = Options::RowMajor>
auto g_rotateLhY(T angle) -> Matrix<T, 4, 4, Option> {
  return g_rotateRhY<T, Option>(-angle);
}

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
 * @param axis The 3D vector representing the axis of rotation.
 * @param angle The rotation angle around the axis, in radians.
 *
 * @note This function normalizes the axis of rotation automatically.
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

template <typename T, Options Option = Options::RowMajor>
auto g_lookAtRh(const Vector3D<T, Option>& eye,
                const Vector3D<T, Option>& target,
                const Vector3D<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto f = target - eye;
  f.normalize();
  auto r = worldUp.cross(f);
  r.normalize();
#else
  auto f = (target - eye).normalize();
  auto r = worldUp.cross(f).normalize();
#endif
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

template <typename T, Options Option = Options::RowMajor>
auto g_lookAtLh(const Vector3D<T, Option>& eye,
                const Vector3D<T, Option>& target,
                const Vector3D<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto f = target - eye;
  f.normalize();
  auto r = worldUp.cross(f);
  r.normalize();
#else
  auto f = (target - eye).normalize();
  auto r = worldUp.cross(f).normalize();
#endif
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

template <typename T, Options Option = Options::RowMajor>
auto g_lookToRh(const Vector3D<T, Option>& eye,
                const Vector3D<T, Option>& direction,
                const Vector3D<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto f = direction;
  f.normalize();
  auto r = worldUp.cross(f);
  r.normalize();
#else
  auto f = direction.normalize();
  auto r = worldUp.cross(f).normalize();
#endif
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

template <typename T, Options Option = Options::RowMajor>
auto g_lookToLh(const Vector3D<T, Option>& eye,
                const Vector3D<T, Option>& direction,
                const Vector3D<T, Option>& worldUp) -> Matrix<T, 4, 4, Option> {
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  auto f = direction;
  f.normalize();
  auto r = worldUp.cross(f);
  r.normalize();
#else
  auto f = direction.normalize();
  auto r = worldUp.cross(f).normalize();
#endif
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
 * Generates a right-handed perspective projection matrix with a depth range of
 * negative one to one.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
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
 * Generates a right-handed perspective projection matrix with a depth range of
 * zero to one.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
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
 * Generates a left-handed perspective projection matrix with a depth range of
 * negative one to one.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
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
 * Generates a left-handed perspective projection matrix with a depth range of
 * zero to one.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
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
 * Generates a right-handed perspective projection matrix based on field of
 * view, width, and height with a depth range of negative one to one.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhNo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveRhNo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * Generates a right-handed perspective projection matrix based on field of
 * view, width, and height with a depth range of zero to one.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveRhZo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveRhZo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * Generates a left-handed perspective projection matrix based on field of view,
 * width, and height with a depth range of negative one to one.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhNo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveLhNo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * Generates a left-handed perspective projection matrix based on field of view,
 * width, and height with a depth range of zero to one.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
 */
template <typename T, Options Option = Options::RowMajor>
auto g_perspectiveLhZo(T fovY, T width, T height, T zNear, T zFar)
    -> Matrix<T, 4, 4, Option> {
  auto aspectRatio       = width / height;
  auto perspectiveMatrix = g_perspectiveLhZo(fovY, aspectRatio, zNear, zFar);
  return perspectiveMatrix;
}

/**
 * Generates a right-handed perspective projection matrix optimized for
 * rendering scenes with an infinite far plane.
 * @note RH-NO-Inf - Right-Handed, Negative One to One depth range, Infinite far
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
 * Generates a right-handed perspective projection matrix for scenes with an
 * infinite far plane, optimized for a zero to one depth range.
 * @note RH-ZO-Inf - Right-Handed, Zero to One depth range, Infinite far plane.
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
 * Generates a left-handed perspective projection matrix for rendering with an
 * infinite far plane, using a depth range of negative one to one.
 * @note LH-NO-Inf - Left-Handed, Negative One to One depth range, Infinite far
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
 * Produces a left-handed perspective projection matrix optimized for scenes
 * with an infinite far plane, and a depth range of zero to one.
 * @note LH-ZO-Inf - Left-Handed, Zero to One depth range, Infinite far plane.
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
 * @param tolerance The tolerance within which the w component is considered
 * effectively zero, preventing division in cases where it might lead to
 * numerical instability. This additional parameter is crucial for handling
 * floating-point inaccuracies, as dividing by values very close to zero can
 * result in Infinity or NaN values. The tolerance allows for a margin of error
 * in floating-point comparisons, ensuring that the division only occurs when
 * the w component is significantly different from zero.
 *
 * @tparam T A floating-point type representing the vector's element type.
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

// BEGIN: frustrum (perspective projection matrix that off center) creation
// functions
// ----------------------------------------------------------------------------

/**
 * Generates a right-handed frustum projection matrix with a depth range of zero
 * to one.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
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
 * Generates a right-handed frustum projection matrix with a depth range of
 * negative one to one.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
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
 * Generates a left-handed frustum projection matrix with a depth range of zero
 * to one.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
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
 * Generates a left-handed frustum projection matrix with a depth range of
 * negative one to one.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
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

// END: frustrum (perspective projection matrix that off center) creation
// functions
// ----------------------------------------------------------------------------

// BEGIN: orthographic projection creation matrix
// ----------------------------------------------------------------------------

// TODO: add ortho functions (LH/RH) that takes (left, right, bottom, top) w/o
// near / far

/**
 * Generates a left-handed orthographic projection matrix with a depth range of
 * zero to one.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
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
 * Generates a left-handed orthographic projection matrix with a depth range of
 * negative one to one.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
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
 * Generates a right-handed orthographic projection matrix with a depth range
 * of zero to one.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
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
 * Generates a right-handed orthographic projection matrix with a depth range
 * of negative one to one.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
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
 * Generates a left-handed orthographic projection matrix based on the given
 * width and height, with a depth range of zero to one. This simplifies setting
 * up the projection by directly specifying the viewport dimensions and the near
 * and far clipping planes.
 * @note LH-ZO - Left-Handed, Zero to One depth range.
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
 * Generates a left-handed orthographic projection matrix based on the given
 * width and height, with a depth range of negative one to one. This simplifies
 * setting up the projection by directly specifying the viewport dimensions and
 * the near and far clipping planes.
 * @note LH-NO - Left-Handed, Negative One to One depth range.
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
 * Generates a right-handed orthographic projection matrix based on the given
 * width and height, with a depth range of zero to one. This simplifies setting
 * up the projection by directly specifying the viewport dimensions and the near
 * and far clipping planes.
 * @note RH-ZO - Right-Handed, Zero to One depth range.
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
 * Generates a right-handed orthographic projection matrix based on the given
 * width and height, with a depth range of negative one to one. This simplifies
 * setting up the projection by directly specifying the viewport dimensions and
 * the near and far clipping planes.
 * @note RH-NO - Right-Handed, Negative One to One depth range.
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
 * @param point The point to be transformed. When a Vector object is passed, it
 * is treated as a point with the homogeneous coordinate set to 1.
 *
 * @note This function automatically applies perspective division for points
 * transformed with a perspective projection matrix. The default tolerance is
 * used for g_perspectiveDivide. Consider adding a parameter to adjust this if
 * needed.
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
  return result.resizedCopy<3>();
}

template <typename T, Options Option = Options::RowMajor>
Vector<T, 3, Option> g_transformVector(const Vector<T, 3, Option>&    vector,
                                       const Matrix<T, 4, 4, Option>& matrix) {
  auto result  = Vector<T, 4, Option>(vector, 0);
  result      *= matrix;
  return result.resizedCopy<3>();
}

// BEGIN: global util vector objects
// ----------------------------------------------------------------------------

// TODO: consider moving directional vector variables to a separate file

// TODO: make these constexpr

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_upVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.y() = 1;
  return vec;
}

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_downVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.y() = -1;
  return vec;
}

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_rightVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.x() = 1;
  return vec;
}

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 2>
auto g_leftVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.x() = -1;
  return vec;
}

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 3>
auto g_forwardVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.z() = 1;
  return vec;
}

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
  requires ValueAtLeast<Size, 3>
auto g_backwardVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(0);
  vec.z() = -1;
  return vec;
}

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
auto g_zeroVector() -> const Vector<T, Size, Option> {
  return Vector<T, Size, Option>(0);
}

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
auto g_oneVector() -> const Vector<T, Size, Option> {
  return Vector<T, Size, Option>(1);
}

template <typename T, std::size_t Size, Options Option = Options::RowMajor>
auto g_unitVector() -> const Vector<T, Size, Option> {
  Vector<T, Size, Option> vec(1);
#ifdef MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
  vec.normalize();
  return vec;
#else
  return vec.normalize();
#endif  // MATH_LIBRARY_USE_NORMALIZE_IN_PLACE
}

// END: global util vector objects
// ----------------------------------------------------------------------------

}  // namespace math

#endif