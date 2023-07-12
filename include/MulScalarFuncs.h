/**
 * @file MulScalarFuncs.h
 */


#pragma once
//
//#include "InstructionSet.h"
//
//namespace math
//{
//
//	template <typename T>
//	typename InstructionSet<T>::MulScalarFunc InstructionSet<T>::getMulScalarFunc()
//	{
//#ifdef SUPPORTS_AVX2
//		return mul_scalar_avx2;
//#elif defined(SUPPORTS_AVX)
//		return mul_scalar_avx;
//#elif defined(SUPPORTS_SSE4_2)
//		return mul_scalar_sse4_2;
//#elif defined(SUPPORTS_SSE4_1)
//		return mul_scalar_sse4_1;
//#elif defined(SUPPORTS_SSSE3)
//		return mul_scalar_ssse3;
//#elif defined(SUPPORTS_SSE3)
//		return mul_scalar_sse3;
//#else
//		return mul_scalar_fallback;
//#endif
//	}
//}