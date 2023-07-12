/**
 * @file AddFuncs.h
 */


//#pragma once
//
//#include "SIMDdefines.h"
//#include <immintrin.h>
//
//#include "InstructionSet.h"
//
//namespace math
//{
//
//	template <>
//	InstructionSet<float>::AddFunc InstructionSet<float>::getAddFunc()
//	{
//#ifdef SUPPORTS_AVX2
//		return add_avx2;
//#elif defined(SUPPORTS_AVX)
//		return add_avx;
//#elif defined(SUPPORTS_SSE4_2)
//		return add_sse4_2;
//#elif defined(SUPPORTS_SSE4_1)
//		return add_sse4_1;
//#elif defined(SUPPORTS_SSSE3)
//		return add_ssse3;
//#elif defined(SUPPORTS_SSE3)
//		return add_sse3;
//#else
//		return add_fallback;
//#endif
//	}
//
//
//
//	template <>
//	void InstructionSet<float>::add_avx2(float* a, const float* b, size_t size)
//	{
//		size_t aligned_size = (size / AVX_SIMD_WIDTH) * AVX_SIMD_WIDTH;
//		size_t i = 0;
//
//		for (i = 0; i < aligned_size; i += AVX_SIMD_WIDTH) {
//			__m256 ymm1 = _mm256_loadu_ps(a + i);
//			__m256 ymm2 = _mm256_loadu_ps(b + i);
//			ymm1 = _mm256_add_ps(ymm1, ymm2);
//			_mm256_storeu_ps(a + i, ymm1);
//		}
//
//		// Handling remaining elements
//		for (; i < size; ++i) {
//			a[i] += b[i];
//		}
//	}
//}