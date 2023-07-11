/**
 * @file AddScalarFuncs.h
 */


#pragma once

#include "InstructionSet.h"

namespace math
{
	template <typename T>
	typename InstructionSet<T>::AddScalarFunc InstructionSet<T>::getAddScalarFunc()
	{
#ifdef SUPPORTS_AVX2
		return add_scalar_avx2;
#elif defined(SUPPORTS_AVX)
		return add_scalar_avx;
#elif defined(SUPPORTS_SSE4_2)
		return add_scalar_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
		return add_scalar_sse4_1;
#elif defined(SUPPORTS_SSSE3)
		return add_scalar_ssse3;
#elif defined(SUPPORTS_SSE3)
		return add_scalar_sse3;
#else
		return add_scalar_fallback;
#endif
	}
}