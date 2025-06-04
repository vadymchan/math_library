
#include "benchmarks.h"
#include "tests.h"

#include <math_library/all.h>

auto main(int argc, char** argv) -> int {
  ::testing::InitGoogleTest(&argc, argv);
  int test_result = RUN_ALL_TESTS();
  if (test_result != 0) {
    return test_result;
  }

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}

/**** QuatTypes.h - Basic type declarations ****/
#ifndef _H_QuatTypes
  #define _H_QuatTypes

/*** Definitions ***/
typedef struct {
  float x, y, z, w;
} Quat; /* Quaternion */

enum QuatPart {
  X,
  Y,
  Z,
  W
};

typedef float HMatrix[4][4]; /* Right-handed, for column vectors */
typedef Quat  EulerAngles;   /* (x,y,z)=ang 1,2,3, w=order code  */
#endif
/**** EOF ****/

/**** EulerAngles.h - Support for 24 angle schemes ****/
/* Ken Shoemake, 1993 */
#ifndef _H_EulerAngles
  #define _H_EulerAngles
// #include "QuatTypes.h"
/*** Order type constants, constructors, extractors ***/
/* There are 24 possible conventions, designated by:    */
/*	  o EulAxI = axis used initially		    */
/*	  o EulPar = parity of axis permutation		    */
/*	  o EulRep = repetition of initial axis as last	    */
/*	  o EulFrm = frame from which axes are taken	    */
/* Axes I,J,K will be a permutation of X,Y,Z.	    */
/* Axis H will be either I or K, depending on EulRep.   */
/* Frame S takes axes from initial static frame.	    */
/* If ord = (AxI=X, Par=Even, Rep=No, Frm=S), then	    */
/* {a,b,c,ord} means Rz(c)Ry(b)Rx(a), where Rz(c)v	    */
/* rotates v around Z by c radians.			    */
  #define EulFrmS     0
  #define EulFrmR     1
  #define EulFrm(ord) ((unsigned)(ord) & 1)
  #define EulRepNo    0
  #define EulRepYes   1
  #define EulRep(ord) (((unsigned)(ord) >> 1) & 1)
  #define EulParEven  0
  #define EulParOdd   1
  #define EulPar(ord) (((unsigned)(ord) >> 2) & 1)
  /* this code is merely a quick (and legal!) way to set arrays, EulSafe being
   * 0,1,2,0 */
  #define EulSafe     "\000\001\002\000"
  #define EulNext     "\001\002\000\001"
  #define EulAxI(ord) ((int)(EulSafe[(((unsigned)(ord) >> 3) & 3)]))
  #define EulAxJ(ord) ((int)(EulNext[EulAxI(ord) + (EulPar(ord) == EulParOdd)]))

  #define EulAxK(ord) ((int)(EulNext[EulAxI(ord) + (EulPar(ord) != EulParOdd)]))

  #define EulAxH(ord) ((EulRep(ord) == EulRepNo) ? EulAxK(ord) : EulAxI(ord))
  /* EulGetOrd unpacks all useful information about order simultaneously. */
  #define EulGetOrd(ord, i, j, k, h, n, s, f) \
    {                                         \
      unsigned o   = (unsigned)ord;           \
      f            = o & 1;                   \
      o          >>= 1;                       \
      s            = o & 1;                   \
      o          >>= 1;                       \
      n            = o & 1;                   \
      o          >>= 1;                       \
      i            = EulSafe[o & 3];          \
      j            = EulNext[i + n];          \
      k            = EulNext[i + 1 - n];      \
      h            = s ? k : i;               \
    }
  /* EulOrd creates an order value between 0 and 23 from 4-tuple choices. */
  #define EulOrd(i, p, r, f) (((((((i) << 1) + (p)) << 1) + (r)) << 1) + (f))
  /* Static axes */
  #define EulOrdXYZs         EulOrd(X, EulParEven, EulRepNo, EulFrmS)
  #define EulOrdXYXs         EulOrd(X, EulParEven, EulRepYes, EulFrmS)
  #define EulOrdXZYs         EulOrd(X, EulParOdd, EulRepNo, EulFrmS)
  #define EulOrdXZXs         EulOrd(X, EulParOdd, EulRepYes, EulFrmS)
  #define EulOrdYZXs         EulOrd(Y, EulParEven, EulRepNo, EulFrmS)
  #define EulOrdYZYs         EulOrd(Y, EulParEven, EulRepYes, EulFrmS)
  #define EulOrdYXZs         EulOrd(Y, EulParOdd, EulRepNo, EulFrmS)
  #define EulOrdYXYs         EulOrd(Y, EulParOdd, EulRepYes, EulFrmS)
  #define EulOrdZXYs         EulOrd(Z, EulParEven, EulRepNo, EulFrmS)
  #define EulOrdZXZs         EulOrd(Z, EulParEven, EulRepYes, EulFrmS)
  #define EulOrdZYXs         EulOrd(Z, EulParOdd, EulRepNo, EulFrmS)
  #define EulOrdZYZs         EulOrd(Z, EulParOdd, EulRepYes, EulFrmS)
  /* Rotating axes */
  #define EulOrdZYXr         EulOrd(X, EulParEven, EulRepNo, EulFrmR)
  #define EulOrdXYXr         EulOrd(X, EulParEven, EulRepYes, EulFrmR)
  #define EulOrdYZXr         EulOrd(X, EulParOdd, EulRepNo, EulFrmR)
  #define EulOrdXZXr         EulOrd(X, EulParOdd, EulRepYes, EulFrmR)
  #define EulOrdXZYr         EulOrd(Y, EulParEven, EulRepNo, EulFrmR)
  #define EulOrdYZYr         EulOrd(Y, EulParEven, EulRepYes, EulFrmR)
  #define EulOrdZXYr         EulOrd(Y, EulParOdd, EulRepNo, EulFrmR)
  #define EulOrdYXYr         EulOrd(Y, EulParOdd, EulRepYes, EulFrmR)
  #define EulOrdYXZr         EulOrd(Z, EulParEven, EulRepNo, EulFrmR)
  #define EulOrdZXZr         EulOrd(Z, EulParEven, EulRepYes, EulFrmR)
  #define EulOrdXYZr         EulOrd(Z, EulParOdd, EulRepNo, EulFrmR)
  #define EulOrdZYZr         EulOrd(Z, EulParOdd, EulRepYes, EulFrmR)

EulerAngles Eul_(float ai, float aj, float ah, int order);
Quat        Eul_ToQuat(EulerAngles ea);
void        Eul_ToHMatrix(EulerAngles ea, HMatrix M);
EulerAngles Eul_FromHMatrix(HMatrix M, int order);
EulerAngles Eul_FromQuat(Quat q, int order);
#endif
/**** EOF ****/

/**** EulerAngles.c - Convert Euler angles to/from matrix or quat ****/
/* Ken Shoemake, 1993 */
// #include "EulerAngles.h"

#include <float.h>
#include <math.h>

EulerAngles Eul_(float ai, float aj, float ah, int order) {
  EulerAngles ea;
  ea.x = ai;
  ea.y = aj;
  ea.z = ah;
  ea.w = (float)order;
  return ea;
}

/* Construct quaternion from Euler angles (in radians). */
Quat Eul_ToQuat(EulerAngles ea) {
  Quat  qu;
  float a[3], ti, tj, th, ci, cj, ch, si, sj, sh, cc, cs, sc, ss;
  int   i, j, k, h, n, s, f;
  EulGetOrd(ea.w, i, j, k, h, n, s, f);
  if (f == EulFrmR) {
    float t = ea.x;
    ea.x    = ea.z;
    ea.z    = t;
  }
  if (n == EulParOdd) {
    ea.y = -ea.y;
  }
  ti = ea.x * 0.5f;
  tj = ea.y * 0.5f;
  th = ea.z * 0.5f;
  ci = cosf(ti);
  cj = cosf(tj);
  ch = cosf(th);
  si = sinf(ti);
  sj = sinf(tj);
  sh = sinf(th);
  cc = ci * ch;
  cs = ci * sh;
  sc = si * ch;
  ss = si * sh;
  if (s == EulRepYes) {
    a[i] = cj * (cs + sc); /* Could speed up with */
    a[j] = sj * (cc + ss); /* trig identities. */
    a[k] = sj * (cs - sc);
    qu.w = cj * (cc - ss);
  } else {
    a[i] = cj * sc - sj * cs;
    a[j] = cj * ss + sj * cc;
    a[k] = cj * cs - sj * sc;
    qu.w = cj * cc + sj * ss;
  }
  if (n == EulParOdd) {
    a[j] = -a[j];
  }
  qu.x = a[X];
  qu.y = a[Y];
  qu.z = a[Z];
  return qu;
}

/* Construct matrix from Euler angles (in radians). */
void Eul_ToHMatrix(EulerAngles ea, HMatrix M) {
  float ti, tj, th, ci, cj, ch, si, sj, sh, cc, cs, sc, ss;
  int   i, j, k, h, n, s, f;
  EulGetOrd(ea.w, i, j, k, h, n, s, f);
  if (f == EulFrmR) {
    float t = ea.x;
    ea.x    = ea.z;
    ea.z    = t;
  }
  if (n == EulParOdd) {
    ea.x = -ea.x;
    ea.y = -ea.y;
    ea.z = -ea.z;
  }
  ti = ea.x;
  tj = ea.y;
  th = ea.z;
  ci = cosf(ti);
  cj = cosf(tj);
  ch = cosf(th);
  si = sinf(ti);
  sj = sinf(tj);
  sh = sinf(th);
  cc = ci * ch;
  cs = ci * sh;
  sc = si * ch;
  ss = si * sh;
  if (s == EulRepYes) {
    M[i][i] = cj;
    M[i][j] = sj * si;
    M[i][k] = sj * ci;
    M[j][i] = sj * sh;
    M[j][j] = -cj * ss + cc;
    M[j][k] = -cj * cs - sc;
    M[k][i] = -sj * ch;
    M[k][j] = cj * sc + cs;
    M[k][k] = cj * cc - ss;
  } else {
    M[i][i] = cj * ch;
    M[i][j] = sj * sc - cs;
    M[i][k] = sj * cc + ss;
    M[j][i] = cj * sh;
    M[j][j] = sj * ss + cc;
    M[j][k] = sj * cs - sc;
    M[k][i] = -sj;
    M[k][j] = cj * si;
    M[k][k] = cj * ci;
  }
  M[W][X] = M[W][Y] = M[W][Z] = M[X][W] = M[Y][W] = M[Z][W] = 0.f;
  M[W][W]                                                   = 1.f;
}

/* Convert matrix to Euler angles (in radians). */
EulerAngles Eul_FromHMatrix(HMatrix M, int order) {
  EulerAngles ea;
  int         i, j, k, h, n, s, f;
  EulGetOrd(order, i, j, k, h, n, s, f);
  if (s == EulRepYes) {
    float sy = sqrtf(M[i][j] * M[i][j] + M[i][k] * M[i][k]);
    if (sy > 16 * FLT_EPSILON) {
      ea.x = atan2f(M[i][j], M[i][k]);
      ea.y = atan2f(sy, M[i][i]);
      ea.z = atan2f(M[j][i], -M[k][i]);
    } else {
      ea.x = atan2f(-M[j][k], M[j][j]);
      ea.y = atan2f(sy, M[i][i]);
      ea.z = 0;
    }
  } else {
    float cy = sqrtf(M[i][i] * M[i][i] + M[j][i] * M[j][i]);
    if (cy > 16 * FLT_EPSILON) {
      ea.x = atan2f(M[k][j], M[k][k]);
      ea.y = atan2f(-M[k][i], cy);
      ea.z = atan2f(M[j][i], M[i][i]);
    } else {
      ea.x = atan2f(-M[j][k], M[j][j]);
      ea.y = atan2f(-M[k][i], cy);
      ea.z = 0;
    }
  }
  if (n == EulParOdd) {
    ea.x = -ea.x;
    ea.y = -ea.y;
    ea.z = -ea.z;
  }
  if (f == EulFrmR) {
    float t = ea.x;
    ea.x    = ea.z;
    ea.z    = t;
  }
  ea.w = (float)order;
  return ea;
}

/* Convert quaternion to Euler angles (in radians). */
EulerAngles Eul_FromQuat(Quat q, int order) {
  HMatrix M;
  float   Nq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  float   s  = (Nq > 0.f) ? (2.f / Nq) : 0.f;
  float   xs = q.x * s, ys = q.y * s, zs = q.z * s;
  float   wx = q.w * xs, wy = q.w * ys, wz = q.w * zs;
  float   xx = q.x * xs, xy = q.x * ys, xz = q.x * zs;
  float   yy = q.y * ys, yz = q.y * zs, zz = q.z * zs;
  M[X][X] = 1.f - (yy + zz);
  M[X][Y] = xy - wz;
  M[X][Z] = xz + wy;
  M[Y][X] = xy + wz;
  M[Y][Y] = 1.f - (xx + zz);
  M[Y][Z] = yz - wx;
  M[Z][X] = xz - wy;
  M[Z][Y] = yz + wx;
  M[Z][Z] = 1.f - (xx + yy);
  M[W][X] = M[W][Y] = M[W][Z] = M[X][W] = M[Y][W] = M[Z][W] = 0.f;
  M[W][W]                                                   = 1.f;
  return Eul_FromHMatrix(M, order);
}

///**** EOF ****/
//
///* EulerSample.c - Read angles as quantum mechanics, write as aerospace */
//// #include "EulerAngles.h"
//
// #include <math_library/quaternion.h>
// #include <stdio.h>
//
// #include <array>
// #include <iostream>
//
// std::string GetRotationOrderString(int order) {
//  switch (order) {
//    // Static frames
//    case EulOrdXYZs:
//      return "EulOrdXYZs";
//    case EulOrdZYXs:
//      return "EulOrdZYXs";
//    case EulOrdZXYs:
//      return "EulOrdZXYs";
//    case EulOrdYXZs:
//      return "EulOrdYXZs";
//    case EulOrdYZXs:
//      return "EulOrdYZXs";
//    case EulOrdXZYs:
//      return "EulOrdXZYs";
//    case EulOrdXYXs:
//      return "EulOrdXYXs";
//    case EulOrdXZXs:
//      return "EulOrdXZXs";
//    case EulOrdYXYs:
//      return "EulOrdYXYs";
//    case EulOrdYZYs:
//      return "EulOrdYZYs";
//    case EulOrdZXZs:
//      return "EulOrdZXZs";
//    case EulOrdZYZs:
//      return "EulOrdZYZs";
//
//    // Dynamic frames
//    case EulOrdXYZr:
//      return "EulOrdXYZr";
//    case EulOrdZYXr:
//      return "EulOrdZYXr";
//    case EulOrdZXYr:
//      return "EulOrdZXYr";
//    case EulOrdYXZr:
//      return "EulOrdYXZr";
//    case EulOrdYZXr:
//      return "EulOrdYZXr";
//    case EulOrdXZYr:
//      return "EulOrdXZYr";
//    case EulOrdXYXr:
//      return "EulOrdXYXr";
//    case EulOrdXZXr:
//      return "EulOrdXZXr";
//    case EulOrdYXYr:
//      return "EulOrdYXYr";
//    case EulOrdYZYr:
//      return "EulOrdYZYr";
//    case EulOrdZXZr:
//      return "EulOrdZXZr";
//    case EulOrdZYZr:
//      return "EulOrdZYZr";
//
//    default:
//      return "Unknown Order";
//  }
//}
//
// void ProcessRotationOrders(const std::array<int, 12>&   rotationOrders,
//                           const std::array<double, 3>& angles) {
//  for (int order : rotationOrders) {
//    std::string orderString = GetRotationOrderString(order);
//    std::cout << "Rotation Order: " << orderString << std::endl;
//    std::cout << "----------------------------------\n";
//
//    for (int i = 0; i < 3; ++i) {
//      for (int j = 0; j < 3; ++j) {
//        if (i == j) {
//          continue;  // Avoid duplicate angles
//        }
//        for (int k = 0; k < 3; ++k) {
//          if (k == i || k == j) {
//            continue;
//          }
//
//          // Log the current angles being used
//          std::cout << "Current Angles: ";
//          std::cout << "angle1 = " << angles[i] << ", ";
//          std::cout << "angle2 = " << angles[j] << ", ";
//          std::cout << "angle3 = " << angles[k] << std::endl;
//
//          // Assign permuted Euler angles with the current rotation order
//          EulerAngles inAngs = {angles[i], angles[j], angles[k], order};
//
//          // Convert Euler angles to Quaternion
//          auto q = Eul_ToQuat(inAngs);
//
//          // Output the quaternion
//          printf("Quat: X Y Z W ");
//          printf("%6.6f  %6.6f  %6.6f  %6.6f\n", q.x, q.y, q.z, q.w);
//
//          // Convert Quaternion to Euler angles
//          EulerAngles outAngs = Eul_FromQuat(q, order);
//
//          // Output the Euler angles
//          printf(" Roll (Pitch)    Pitch (Yaw)   Yaw (Roll)    (radians)\n");
//          printf("%6.6f            %6.6f         %6.6f\n",
//                 outAngs.x,
//                 outAngs.y,
//                 outAngs.z);
//          printf("----------------------------------\n");
//        }
//      }
//    }
//    std::cout << std::endl;
//  }
//}
//
// void main(void) {
//  // EulerAngles outAngs, inAngs = {0, 0, 0, EulOrdZYXs};
//  // HMatrix     R;
//
//  // printf("Phi Theta Psi (radians): ");
//  // scanf("%f %f %f", &inAngs.x, &inAngs.y, &inAngs.z);
//
//  // Eul_ToHMatrix(inAngs, R);
//
//  // outAngs = Eul_FromHMatrix(R, EulOrdZYXs);
//  //
//  // printf(" Roll    Pitch   Yaw    (radians)\n");
//  // printf("%6.3f  %6.3f  %6.3f\n", outAngs.x, outAngs.y, outAngs.z);
//
//  EulerAngles outAngs, inAngs = {0, 0, 0, EulOrdXYZs};
//  // inAngs = {0, 0, 0, EulOrdZYXs};
//  //  inAngs = {0, 0, 0, EulOrdZXYs};
//  //  inAngs = {0, 0, 0, EulOrdYXZs};
//  //  inAngs = {0, 0, 0, EulOrdYZXs};
//  //  inAngs = {0, 0, 0, EulOrdXZYs};
//  //  inAngs = {0, 0, 0, EulOrdXYXs};
//  //  inAngs = {0, 0, 0, EulOrdXZXs};
//  //  inAngs = {0, 0, 0, EulOrdYXYs};
//  //  inAngs = {0, 0, 0, EulOrdYZYs};
//  //  inAngs = {0, 0, 0, EulOrdZXZs};
//  //  inAngs = {0, 0, 0, EulOrdZYZs};
//
//  // EulerAngles inAngs1 = {0, 0, 0, EulOrdXYZs};
//
//  // inAngs.x = math::g_kPi / 6;
//  // inAngs.y = math::g_kPi / 3;
//  // inAngs.z = math::g_kPi / 4;
//
//  // auto q = Eul_ToQuat(inAngs);
//
//  // printf("Quat: X Y Z W ");
//  // printf("%6.3f  %6.3f  %6.3f  %6.3f\n", q.x, q.y, q.z, q.w);
//
//  // outAngs = Eul_FromQuat(q, EulOrdZYXs);
//
//  // printf(" Roll (Pitch)    Pitch (Yaw)   Yaw (Roll)    (radians)\n");
//  // printf("%6.3f  %6.3f  %6.3f\n", outAngs.x, outAngs.y, outAngs.z);
//
//  // Quat q;
//
//  // printf("Quat: X Y Z W ");
//  // scanf("%f %f %f %f", &q.x, &q.y, &q.z, &q.w);
//
//  // EulerAngles outAngs = Eul_FromQuat(q, EulOrdXYZs);
//
//  // printf(" Roll (Pitch)    Pitch (Yaw)   Yaw (Roll)    (radians)\n");
//  // printf("%6.3f            %6.3f         %6.3f\n",
//  //        outAngs.x,
//  //        outAngs.y,
//  //        outAngs.z);
//
//  // printf("----------------------------------");
//
//  const std::array<int, 12> rotationOrdersStatic = {EulOrdXYZs,
//                                                    EulOrdZYXs,
//                                                    EulOrdZXYs,
//                                                    EulOrdYXZs,
//                                                    EulOrdYZXs,
//                                                    EulOrdXZYs,
//                                                    EulOrdXYXs,
//                                                    EulOrdXZXs,
//                                                    EulOrdYXYs,
//                                                    EulOrdYZYs,
//                                                    EulOrdZXZs,
//                                                    EulOrdZYZs};
//
//  const std::array<int, 12> rotationOrdersDynamic = {EulOrdXYZr,
//                                                     EulOrdZYXr,
//                                                     EulOrdZXYr,
//                                                     EulOrdYXZr,
//                                                     EulOrdYZXr,
//                                                     EulOrdXZYr,
//                                                     EulOrdXYXr,
//                                                     EulOrdXZXr,
//                                                     EulOrdYXYr,
//                                                     EulOrdYZYr,
//                                                     EulOrdZXZr,
//                                                     EulOrdZYZr};
//
//  const std::array<double, 3> angles
//      = {math::g_kPi / 6, math::g_kPi / 3, math::g_kPi / 4};
//
//  // Process both static and dynamic frames
//  std::cout << "Processing Static Frames:" << std::endl;
//  ProcessRotationOrders(rotationOrdersStatic, angles);
//  std::cout << "Processing Dynamic Frames:" << std::endl;
//  ProcessRotationOrders(rotationOrdersDynamic, angles);
//}

// ----------------------------------------------------------

#include <math_library/quaternion.h>

#include <array>
#include <iomanip>

namespace math {

//// Helper function to get the parameters from the rotation order
// void getEulerOrderParameters(EulerRotationOrder order,
//                              Axis&              i,
//                              Axis&              j,
//                              Axis&              k,
//                              Axis&              h,
//                              bool&              parityOdd,
//                              bool&              repetition) {
//   switch (order) {
//     case EulerRotationOrder::XYZ:
//       i          = Axis::X;
//       parityOdd  = false;
//       repetition = false;
//       break;
//     case EulerRotationOrder::XYX:
//       i          = Axis::X;
//       parityOdd  = false;
//       repetition = true;
//       break;
//     case EulerRotationOrder::XZY:
//       i          = Axis::X;
//       parityOdd  = true;
//       repetition = false;
//       break;
//     case EulerRotationOrder::XZX:
//       i          = Axis::X;
//       parityOdd  = true;
//       repetition = true;
//       break;
//     case EulerRotationOrder::YZX:
//       i          = Axis::Y;
//       parityOdd  = false;
//       repetition = false;
//       break;
//     case EulerRotationOrder::YZY:
//       i          = Axis::Y;
//       parityOdd  = false;
//       repetition = true;
//       break;
//     case EulerRotationOrder::YXZ:
//       i          = Axis::Y;
//       parityOdd  = true;
//       repetition = false;
//       break;
//     case EulerRotationOrder::YXY:
//       i          = Axis::Y;
//       parityOdd  = true;
//       repetition = true;
//       break;
//     case EulerRotationOrder::ZXY:
//       i          = Axis::Z;
//       parityOdd  = false;
//       repetition = false;
//       break;
//     case EulerRotationOrder::ZXZ:
//       i          = Axis::Z;
//       parityOdd  = false;
//       repetition = true;
//       break;
//     case EulerRotationOrder::ZYX:
//       i          = Axis::Z;
//       parityOdd  = true;
//       repetition = false;
//       break;
//     case EulerRotationOrder::ZYZ:
//       i          = Axis::Z;
//       parityOdd  = true;
//       repetition = true;
//       break;
//     default:
//       assert(false && "Invalid Euler rotation order");
//       break;
//   }
//
//   int32_t axisI = static_cast<int32_t>(i);
//
//   int32_t axisJ, axisK;
//
//   if (!parityOdd) {
//     axisJ = (axisI + 1) % 3;
//     axisK = (axisI + 2) % 3;
//   } else {
//     axisJ = (axisI + 2) % 3;
//     axisK = (axisI + 1) % 3;
//   }
//
//   j = static_cast<Axis>(axisJ);
//   k = static_cast<Axis>(axisK);
//
//   h = repetition ? i : k;
// }
//
// template <typename T>
// Quaternion<T> fromEulerAngles(
//     T ai, T aj, T ah, EulerRotationOrder order, Frame frame = Frame::Static)
//     {
//   Axis i, j, k, h;
//   bool parityOdd, repetition;
//   getEulerOrderParameters(order, i, j, k, h, parityOdd, repetition);
//
//   if (frame == Frame::Rotating) {
//     std::swap(ai, ah);
//   }
//   if (parityOdd) {
//     aj = -aj;
//   }
//
//   T ti = ai * T(0.5);
//   T tj = aj * T(0.5);
//   T th = ah * T(0.5);
//
//   T ci = std::cos(ti);
//   T cj = std::cos(tj);
//   T ch = std::cos(th);
//
//   T si = std::sin(ti);
//   T sj = std::sin(tj);
//   T sh = std::sin(th);
//
//   T cc = ci * ch;
//   T cs = ci * sh;
//   T sc = si * ch;
//   T ss = si * sh;
//
//   T qx, qy, qz, qw;
//
//   if (repetition) {
//     qx = cj * (cs + sc);
//     qy = sj * (cc + ss);
//     qz = sj * (cs - sc);
//     qw = cj * (cc - ss);
//   } else {
//     qx = cj * sc - sj * cs;
//     qy = cj * ss + sj * cc;
//     qz = cj * cs - sj * sc;
//     qw = cj * cc + sj * ss;
//   }
//
//   if (parityOdd) {
//     qy = -qy;
//   }
//
//   T q[3]                     = {0, 0, 0};
//   q[static_cast<int32_t>(i)] = qx;
//   q[static_cast<int32_t>(j)] = qy;
//   q[static_cast<int32_t>(k)] = qz;
//
//   return Quaternion<T>(q[0], q[1], q[2], qw).normalized();
// }
//
// template <typename T>
// void toEulerAngles(const Quaternion<T>& q,
//                   T&                   angle1,
//                   T&                   angle2,
//                   T&                   angle3,
//                   EulerRotationOrder   order,
//                   Frame                frame = Frame::Static) {
//  // Get rotation order parameters
//  Axis i, j, k, h;
//  bool parityOdd, repetition;
//  getEulerOrderParameters(order, i, j, k, h, parityOdd, repetition);
//
//  int32_t axisI = static_cast<int32_t>(i);
//  int32_t axisJ = static_cast<int32_t>(j);
//  int32_t axisK = static_cast<int32_t>(k);
//
//  // Convert quaternion to rotation matrix
//  T Nq = q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w();
//  T s  = (Nq > T(0)) ? (T(2) / Nq) : T(0);
//
//  T xs = q.x() * s, ys = q.y() * s, zs = q.z() * s;
//  T wx = q.w() * xs, wy = q.w() * ys, wz = q.w() * zs;
//  T xx = q.x() * xs, xy = q.x() * ys, xz = q.x() * zs;
//  T yy = q.y() * ys, yz = q.y() * zs, zz = q.z() * zs;
//
//  std::array<std::array<T, 3>, 3> M;
//
//  M[0][0] = T(1) - (yy + zz);
//  M[0][1] = xy - wz;
//  M[0][2] = xz + wy;
//  M[1][0] = xy + wz;
//  M[1][1] = T(1) - (xx + zz);
//  M[1][2] = yz - wx;
//  M[2][0] = xz - wy;
//  M[2][1] = yz + wx;
//  M[2][2] = T(1) - (xx + yy);
//
//  // Extract Euler angles from the rotation matrix
//  if (repetition) {
//    T sy = std::sqrt(M[axisI][axisJ] * M[axisI][axisJ]
//                     + M[axisI][axisK] * M[axisI][axisK]);
//    if (sy > T(16) * std::numeric_limits<T>::epsilon()) {
//      angle1 = std::atan2(M[axisI][axisJ], M[axisI][axisK]);
//      angle2 = std::atan2(sy, M[axisI][axisI]);
//      angle3 = std::atan2(M[axisJ][axisI], -M[axisK][axisI]);
//    } else {
//      angle1 = std::atan2(-M[axisJ][axisK], M[axisJ][axisJ]);
//      angle2 = std::atan2(sy, M[axisI][axisI]);
//      angle3 = T(0);
//    }
//  } else {
//    T cy = std::sqrt(M[axisI][axisI] * M[axisI][axisI]
//                     + M[axisJ][axisI] * M[axisJ][axisI]);
//    if (cy > T(16) * std::numeric_limits<T>::epsilon()) {
//      angle1 = std::atan2(M[axisK][axisJ], M[axisK][axisK]);
//      angle2 = std::atan2(-M[axisK][axisI], cy);
//      angle3 = std::atan2(M[axisJ][axisI], M[axisI][axisI]);
//    } else {
//      angle1 = std::atan2(-M[axisJ][axisK], M[axisJ][axisJ]);
//      angle2 = std::atan2(-M[axisK][axisI], cy);
//      angle3 = T(0);
//    }
//  }
//
//  if (parityOdd) {
//    angle1 = -angle1;
//    angle2 = -angle2;
//    angle3 = -angle3;
//  }
//
//  if (frame == Frame::Rotating) {
//    std::swap(angle1, angle3);
//  }
//}

std::string GetRotationOrderString(EulerRotationOrder order, Frame frame) {
  std::string frameType = (frame == Frame::Static) ? "s" : "r";
  switch (order) {
    case EulerRotationOrder::XYZ:
      return "EulOrdXYZ" + frameType;
    case EulerRotationOrder::XYX:
      return "EulOrdXYX" + frameType;
    case EulerRotationOrder::XZY:
      return "EulOrdXZY" + frameType;
    case EulerRotationOrder::XZX:
      return "EulOrdXZX" + frameType;
    case EulerRotationOrder::YZX:
      return "EulOrdYZX" + frameType;
    case EulerRotationOrder::YZY:
      return "EulOrdYZY" + frameType;
    case EulerRotationOrder::YXZ:
      return "EulOrdYXZ" + frameType;
    case EulerRotationOrder::YXY:
      return "EulOrdYXY" + frameType;
    case EulerRotationOrder::ZXY:
      return "EulOrdZXY" + frameType;
    case EulerRotationOrder::ZXZ:
      return "EulOrdZXZ" + frameType;
    case EulerRotationOrder::ZYX:
      return "EulOrdZYX" + frameType;
    case EulerRotationOrder::ZYZ:
      return "EulOrdZYZ" + frameType;
    default:
      return "Unknown";
  }
}

void TestEulerToQuaternion() {
  // Define rotation orders
  const std::array<EulerRotationOrder, 12> rotationOrders
      = {EulerRotationOrder::XYZ,
         EulerRotationOrder::ZYX,
         EulerRotationOrder::ZXY,
         EulerRotationOrder::YXZ,
         EulerRotationOrder::YZX,
         EulerRotationOrder::XZY,
         EulerRotationOrder::XYX,
         EulerRotationOrder::XZX,
         EulerRotationOrder::YXY,
         EulerRotationOrder::YZY,
         EulerRotationOrder::ZXZ,
         EulerRotationOrder::ZYZ};

  // Define test angles (in radians)
  const std::array<double, 3> angles = {
    math::g_kPi / 6,  // 30 degrees
    math::g_kPi / 3,  // 60 degrees
    math::g_kPi / 4   // 45 degrees
  };

  // Define frame types: Static and Dynamic
  const std::array<Frame, 2> frames = {Frame::Static, Frame::Rotating};

  // Loop over frame types
  for (Frame frame : frames) {
    std::string frameType = (frame == Frame::Static) ? "Static" : "Dynamic";
    std::cout << "Testing for Frame: " << frameType << std::endl;

    // Loop over rotation orders
    for (EulerRotationOrder order : rotationOrders) {
      std::string orderString = GetRotationOrderString(order, frame);
      std::cout << "Rotation Order: " << orderString << std::endl;
      std::cout << "----------------------------------\n";

      // Loop over permutations of angles
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (i == j) {
            continue;  // Avoid duplicate angles
          }
          for (int k = 0; k < 3; ++k) {
            if (k == i || k == j) {
              continue;
            }

            // Log the current angles being used
            std::cout << "Current Angles: ";
            std::cout << "angle1 = " << angles[i] << ", ";
            std::cout << "angle2 = " << angles[j] << ", ";
            std::cout << "angle3 = " << angles[k] << std::endl;

            // Assign permuted Euler angles and convert them to a quaternion
            double pitch = angles[i];
            double yaw   = angles[j];
            double roll  = angles[k];

            // Convert Euler angles to Quaternion
            auto q = Quaternion<double>::fromEulerAngles(
                pitch, yaw, roll, order, frame);

            // Output the quaternion
            std::cout << "Quat: X Y Z W " << std::fixed << std::setprecision(6)
                      << q.x() << "  " << q.y() << "  " << q.z() << "  "
                      << q.w() << std::endl;

            // Convert Quaternion back to Euler anglesw
            double pitch_out, yaw_out, roll_out;
            q.toEulerAngles(pitch_out, yaw_out, roll_out, order, frame);

            // Output the Euler angles
            std::cout
                << " Roll (Pitch)    Pitch (Yaw)   Yaw (Roll)    (radians)\n";
            printf("%6.6f            %6.6f         %6.6f\n",
                   pitch_out,
                   yaw_out,
                   roll_out);
            std::cout << "----------------------------------\n";
          }
        }
      }
      std::cout << std::endl;
    }
  }
}

}  // namespace math

#include <iostream>

// Example usage
// int main() {
//  math::TestEulerToQuaternion();
//
//  return 0;
//}
