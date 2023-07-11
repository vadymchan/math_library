/**
 * @file MultiplyFuncs.h
 */


#pragma once


#include "InstructionSet.h"

namespace math
{
	template <typename T>
	typename InstructionSet<T>::MultiplyFunc InstructionSet<T>::getMultiplyFunc()
		{
	#ifdef SUPPORTS_AVX2
			return multiply_avx2;
	#elif defined(SUPPORTS_AVX)
			return multiply_avx;
	#elif defined(SUPPORTS_SSE4_2)
			return multiply_sse4_2;
	#elif defined(SUPPORTS_SSE4_1)
			return multiply_sse4_1;
	#elif defined(SUPPORTS_SSSE3)
			return multiply_ssse3;
	#elif defined(SUPPORTS_SSE3)
			return multiply_sse3;
	#else
			return multiply_fallback;
	#endif
		}
}