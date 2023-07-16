/**
 * @file InstructionSetFloat.h
 */



#pragma once


#include <immintrin.h>


namespace math
{

	template<typename T>
	class InstructionSet;

	template<>
	class InstructionSet<float>
	{
	public:


		using AddFunc = void(*)(float*, const float*, size_t);

		static AddFunc getAddFunc()
		{
#ifdef SUPPORTS_AVX2
			return add_avx2;
#elif defined(SUPPORTS_AVX)
			return add_avx;
#elif defined(SUPPORTS_SSE4_2)
			return add_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
			return add_sse4_1;
#elif defined(SUPPORTS_SSSE3)
			return add_ssse3;
#elif defined(SUPPORTS_SSE3)
			return add_sse3;
#else
			return add_fallback;
#endif
		}

		using AddScalarFunc = void(*)(float*, float, size_t);

		static AddScalarFunc getAddScalarFunc()
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

		using MultiplyFunc = void(*)(float*, const float*, const float*, size_t);

		static MultiplyFunc getMultiplyFunc()
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

		using MulFunc = void(*)(float*, const float*, const float*, size_t);

		static MulFunc getMulFunc()
		{
#ifdef SUPPORTS_AVX2
			return mul_avx2;
#elif defined(SUPPORTS_AVX)
			return mul_avx;
#elif defined(SUPPORTS_SSE4_2)
			return mul_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
			return mul_sse4_1;
#elif defined(SUPPORTS_SSSE3)
			return mul_ssse3;
#elif defined(SUPPORTS_SSE3)
			return mul_sse3;
#else
			return mul_fallback;
#endif
		}

		using MulScalarFunc = void(*)(float*, float, size_t);

		static MulScalarFunc getMulScalarFunc()
		{
#ifdef SUPPORTS_AVX2
			return mul_scalar_avx2;
#elif defined(SUPPORTS_AVX)
			return mul_scalar_avx;
#elif defined(SUPPORTS_SSE4_2)
			return mul_scalar_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
			return mul_scalar_sse4_1;
#elif defined(SUPPORTS_SSSE3)
			return mul_scalar_ssse3;
#elif defined(SUPPORTS_SSE3)
			return mul_scalar_sse3;
#else
			return mul_scalar_fallback;
#endif
		}



	private:

		static constexpr size_t AVX_SIMD_WIDTH = 8;

		//BEGIN: add two arrays
		//----------------------------------------------------------------------------

		static void add_avx2(float* a, const float* b, size_t size)
		{
			size_t aligned_size = (size / AVX_SIMD_WIDTH) * AVX_SIMD_WIDTH;
			size_t i = 0;

			for (i = 0; i < aligned_size; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				__m256 ymm2 = _mm256_loadu_ps(b + i);
				ymm1 = _mm256_add_ps(ymm1, ymm2);
				_mm256_storeu_ps(a + i, ymm1);
			}

			// Handling remaining elements
			for (; i < size; ++i) {
				a[i] += b[i];
			}
		}


		static void add_avx(float* a, const float* b, size_t size)
		{
		}
		static void add_sse4_2(float* a, const float* b, size_t size)
		{
		}
		static void add_sse4_1(float* a, const float* b, size_t size)
		{
		}
		static void add_ssse3(float* a, const float* b, size_t size)
		{
		}
		static void add_sse3(float* a, const float* b, size_t size)
		{
		}
		static void add_fallback(float* a, const float* b, size_t size)
		{
		}

		//END: add two arrays
		//----------------------------------------------------------------------------

		//BEGIN: add scalar
		//----------------------------------------------------------------------------

		static void add_scalar_avx2(float* a, float scalar, size_t size)
		{
			__m256 ymm0 = _mm256_set1_ps(scalar);
			size_t i = 0;

			for (; i < size; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				ymm1 = _mm256_add_ps(ymm1, ymm0);
				_mm256_storeu_ps(a + i, ymm1);
			}

			for (; i < size; ++i) {
				a[i] += scalar;
			}
		}

		static void add_scalar_avx(float* a, float scalar, size_t size)
		{

		}

		static void add_scalar_sse4_2(float* a, float scalar, size_t size)
		{

		}

		static void add_scalar_sse4_1(float* a, float scalar, size_t size)
		{

		}

		static void add_scalar_ssse3(float* a, float scalar, size_t size)
		{

		}

		static void add_scalar_sse3(float* a, float scalar, size_t size)
		{

		}

		static void add_scalar_fallback(float* a, float scalar, size_t size)
		{
			//no SIMD
			for (size_t i = 0; i < size; ++i)
				a[i] += scalar;
		}

		//END: add scalar
		//----------------------------------------------------------------------------


		//BEGIN: multiplication array
		//----------------------------------------------------------------------------

		static void mul_avx2(float* result, const float* a, const float* b, size_t size)
		{
		}

		static void mul_avx(float* result, const float* a, const float* b, size_t size)
		{
		}

		static void mul_sse4_2(float* result, const float* a, const float* b, size_t size)
		{
		}

		static void mul_sse4_1(float* result, const float* a, const float* b, size_t size)
		{
		}

		static void mul_ssse3(float* result, const float* a, const float* b, size_t size)
		{
		}

		static void mul_sse3(float* result, const float* a, const float* b, size_t size)
		{
		}

		static void mul_fallback(float* result, const float* a, const float* b, size_t size)
		{
		}

		//END: multiplication array
		//----------------------------------------------------------------------------



		//BEGIN: multiplication scalar
		//----------------------------------------------------------------------------

		static void mul_scalar_avx2(float* a, float scalar, size_t size)
		{
			
			mul_scalar_avx(a, scalar, size);
		}

		static void mul_scalar_avx(float* a, float scalar, size_t size)
		{
		}

		static void mul_scalar_sse4_2(float* a, float scalar, size_t size)
		{
		}

		static void mul_scalar_sse4_1(float* a, float scalar, size_t size)
		{
		}

		static void mul_scalar_ssse3(float* a, float scalar, size_t size)
		{
		}

		static void mul_scalar_sse3(float* a, float scalar, size_t size)
		{
		}

		static void mul_scalar_fallback(float* a, float scalar, size_t size)
		{
		}

		//END: multiplication scalar
		//----------------------------------------------------------------------------

	};

}