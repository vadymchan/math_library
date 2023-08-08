/**
 * @file InstructionSetFloat.h
 */



#pragma once

#include "../../options/Options.h"
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

		using SubFunc = void(*)(float*, const float*, size_t);

		static SubFunc getSubFunc()
		{
#ifdef SUPPORTS_AVX2
			return sub_avx2;
#elif defined(SUPPORTS_AVX)
			return sub_avx;
#elif defined(SUPPORTS_SSE4_2)
			return sub_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
			return sub_sse4_1;
#elif defined(SUPPORTS_SSSE3)
			return sub_ssse3;
#elif defined(SUPPORTS_SSE3)
			return sub_sse3;
#else
			return sub_fallback;
#endif
		}

		using SubScalarFunc = void(*)(float*, float, size_t);

		static SubScalarFunc getSubScalarFunc()
		{
#ifdef SUPPORTS_AVX2
			return sub_scalar_avx2;
#elif defined(SUPPORTS_AVX)
			return sub_scalar_avx;
#elif defined(SUPPORTS_SSE4_2)
			return sub_scalar_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
			return sub_scalar_sse4_1;
#elif defined(SUPPORTS_SSSE3)
			return sub_scalar_ssse3;
#elif defined(SUPPORTS_SSE3)
			return sub_scalar_sse3;
#else
			return sub_scalar_fallback;
#endif
		}

		template<Options Option>
		using MulFunc = void(*)(float*, const float*, const float*, const size_t, const size_t, const size_t);

		template<Options Option>
		static MulFunc<Option> getMulFunc()
		{

#ifdef SUPPORTS_AVX2
			return mul_avx2<Option>;
#elif defined(SUPPORTS_AVX)
			return mul_avx<Option>;
#elif defined(SUPPORTS_SSE4_2)
			return mul_sse4_2<Option>;
#elif defined(SUPPORTS_SSE4_1)
			return mul_sse4_1<Option>;
#elif defined(SUPPORTS_SSSE3)
			return mul_ssse3<Option>;
#elif defined(SUPPORTS_SSE3)
			return mul_sse3<Option>;
#else
			return mul_fallback<Option>;
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

		using DivScalarFunc = void(*)(float*, float, size_t);

		static DivScalarFunc getDivScalarFunc()
		{
#ifdef SUPPORTS_AVX2
			return div_scalar_avx2;
#elif defined(SUPPORTS_AVX)
			return div_scalar_avx;
#elif defined(SUPPORTS_SSE4_2)
			return div_scalar_sse4_2;
#elif defined(SUPPORTS_SSE4_1)
			return div_scalar_sse4_1;
#elif defined(SUPPORTS_SSSE3)
			return div_scalar_ssse3;
#elif defined(SUPPORTS_SSE3)
			return div_scalar_sse3;
#else
			return div_scalar_fallback;
#endif
		}


	private:

		static constexpr size_t AVX_SIMD_WIDTH = 8;
		static constexpr size_t SSE_SIMD_WIDTH = 4;

		//BEGIN: add two arrays
		//----------------------------------------------------------------------------

		static void add_avx2(float* a, const float* b, size_t size)
		{
			add_avx(a, b, size);
		}


		static void add_avx(float* a, const float* b, size_t size)
		{
			const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
			size_t i = 0;

			for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				__m256 ymm2 = _mm256_loadu_ps(b + i);
				ymm1 = _mm256_add_ps(ymm1, ymm2);
				_mm256_storeu_ps(a + i, ymm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] += b[i];
			}
		}


		static void add_sse4_2(float* a, const float* b, size_t size)
		{
			add_sse3(a, b, size);
		}

		static void add_sse4_1(float* a, const float* b, size_t size)
		{
			add_sse3(a, b, size);
		}

		static void add_ssse3(float* a, const float* b, size_t size)
		{
			add_sse3(a, b, size);
		}

		static void add_sse3(float* a, const float* b, size_t size)
		{
			const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
			size_t i = 0;

			for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				__m128 xmm2 = _mm_loadu_ps(b + i);
				xmm1 = _mm_add_ps(xmm1, xmm2);
				_mm_storeu_ps(a + i, xmm1);
			}

			// Handling remaining elements
			for (; i < size; ++i) {
				a[i] += b[i];
			}
		}


		static void add_fallback(float* a, const float* b, size_t size)
		{
			for (size_t i = 0; i < size; ++i) {
				a[i] += b[i];
			}
		}


		//END: add two arrays
		//----------------------------------------------------------------------------

		//BEGIN: add scalar
		//----------------------------------------------------------------------------

		static void add_scalar_avx2(float* a, float scalar, size_t size)
		{
			add_scalar_avx(a, scalar, size);
		}

		static void add_scalar_avx(float* a, float scalar, size_t size)
		{
			__m256 ymm0 = _mm256_set1_ps(scalar);
			const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
			size_t i = 0;

			for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				ymm1 = _mm256_add_ps(ymm1, ymm0);
				_mm256_storeu_ps(a + i, ymm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] += scalar;
			}
		}


		static void add_scalar_sse4_2(float* a, float scalar, size_t size)
		{
			add_scalar_sse3(a, scalar, size);
		}

		static void add_scalar_sse4_1(float* a, float scalar, size_t size)
		{
			add_scalar_sse3(a, scalar, size);
		}

		static void add_scalar_ssse3(float* a, float scalar, size_t size)
		{
			add_scalar_sse3(a, scalar, size);
		}

		static void add_scalar_sse3(float* a, float scalar, size_t size)
		{
			__m128 xmm0 = _mm_set1_ps(scalar);
			const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
			size_t i = 0;

			for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				xmm1 = _mm_add_ps(xmm1, xmm0);
				_mm_storeu_ps(a + i, xmm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] += scalar;
			}
		}


		static void add_scalar_fallback(float* a, float scalar, size_t size)
		{
			//no SIMD
			for (size_t i = 0; i < size; ++i)
				a[i] += scalar;
		}

		//END: add scalar
		//----------------------------------------------------------------------------

		//BEGIN: subtract two arrays
		//----------------------------------------------------------------------------

		static void sub_avx2(float* a, const float* b, size_t size)
		{
			sub_avx(a, b, size);
		}

		static void sub_avx(float* a, const float* b, size_t size)
		{
			const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
			size_t i = 0;

			for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				__m256 ymm2 = _mm256_loadu_ps(b + i);
				ymm1 = _mm256_sub_ps(ymm1, ymm2);
				_mm256_storeu_ps(a + i, ymm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] -= b[i];
			}
		}


		static void sub_sse4_2(float* a, const float* b, size_t size)
		{
			sub_sse3(a, b, size);
		}

		static void sub_sse4_1(float* a, const float* b, size_t size)
		{
			sub_sse3(a, b, size);
		}

		static void sub_ssse3(float* a, const float* b, size_t size)
		{
			sub_sse3(a, b, size);
		}

		static void sub_sse3(float* a, const float* b, size_t size)
		{
			const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
			size_t i = 0;

			for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				__m128 xmm2 = _mm_loadu_ps(b + i);
				xmm1 = _mm_sub_ps(xmm1, xmm2);
				_mm_storeu_ps(a + i, xmm1);
			}

			// Handle any remaining elements
			for (; i < size; ++i) {
				a[i] -= b[i];
			}
		}


		static void sub_fallback(float* a, const float* b, size_t size)
		{
			for (size_t i = 0; i < size; ++i) {
				a[i] -= b[i];
			}
		}

		//END: subtract two arrays
		//----------------------------------------------------------------------------

		static void sub_scalar_avx2(float* a, float scalar, size_t size)
		{
			sub_scalar_avx(a, scalar, size);
		}

		static void sub_scalar_avx(float* a, float scalar, size_t size)
		{
			__m256 ymm0 = _mm256_set1_ps(scalar);
			const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
			size_t i = 0;

			for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				ymm1 = _mm256_sub_ps(ymm1, ymm0);
				_mm256_storeu_ps(a + i, ymm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] -= scalar;
			}
		}


		static void sub_scalar_sse4_2(float* a, float scalar, size_t size)
		{
			sub_scalar_sse3(a, scalar, size);
		}

		static void sub_scalar_sse4_1(float* a, float scalar, size_t size)
		{
			sub_scalar_sse3(a, scalar, size);
		}

		static void sub_scalar_ssse3(float* a, float scalar, size_t size)
		{
			sub_scalar_sse3(a, scalar, size);
		}

		static void sub_scalar_sse3(float* a, float scalar, size_t size)
		{
			__m128 xmm0 = _mm_set1_ps(scalar);
			const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
			size_t i = 0;

			for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				xmm1 = _mm_sub_ps(xmm1, xmm0);
				_mm_storeu_ps(a + i, xmm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] -= scalar;
			}
		}


		static void sub_scalar_fallback(float* a, float scalar, size_t size)
		{
			for (size_t i = 0; i < size; ++i) {
				a[i] -= scalar;
			}
		}

		//END: subtract scalar
		//----------------------------------------------------------------------------

		//BEGIN: multiplication array
		//----------------------------------------------------------------------------

		//BEGIN: multiplication array utility functions

		template<Options Option>
		static inline size_t indexA(const size_t currentRowA, const size_t innerIndex, const size_t rowsA, const size_t colsA_rowsB)
		{
			if constexpr (Option == Options::ColumnMajor) {
				return currentRowA + innerIndex * rowsA;
			}
			else if constexpr (Option == Options::RowMajor) {
				return currentRowA * colsA_rowsB + innerIndex;
			}
		}

		template<Options Option>
		static inline size_t indexB(const size_t innerIndex, const size_t currentColB, const size_t colsB, const size_t colsA_rowsB)
		{
			if constexpr (Option == Options::ColumnMajor) {
				return innerIndex + currentColB * colsA_rowsB;
			}
			else if constexpr (Option == Options::RowMajor) {
				return innerIndex * colsB + currentColB;
			}
		}

		template<Options Option>
		static inline size_t indexResult(const size_t currentRowA, const size_t currentColB, const size_t rowsA, const size_t colsB)
		{
			if constexpr (Option == Options::ColumnMajor) {
				return currentRowA + currentColB * rowsA;
			}
			else if constexpr (Option == Options::RowMajor) {
				return currentRowA * colsB + currentColB;
			}
		}

		template<Options Option>
		static inline __m256 loadA(const float* a, const size_t currentRowA, const size_t innerIndex, const size_t rowsA, const size_t colsA_rowsB)
		{
			if constexpr (Option == Options::RowMajor) {
				return _mm256_loadu_ps(&a[indexA<Option>(currentRowA, innerIndex, rowsA, colsA_rowsB)]);
			}
			else {
				return _mm256_set_ps(
					a[indexA<Option>(currentRowA, innerIndex + 7, rowsA, colsA_rowsB)],
					a[indexA<Option>(currentRowA, innerIndex + 6, rowsA, colsA_rowsB)],
					a[indexA<Option>(currentRowA, innerIndex + 5, rowsA, colsA_rowsB)],
					a[indexA<Option>(currentRowA, innerIndex + 4, rowsA, colsA_rowsB)],
					a[indexA<Option>(currentRowA, innerIndex + 3, rowsA, colsA_rowsB)],
					a[indexA<Option>(currentRowA, innerIndex + 2, rowsA, colsA_rowsB)],
					a[indexA<Option>(currentRowA, innerIndex + 1, rowsA, colsA_rowsB)],
					a[indexA<Option>(currentRowA, innerIndex, rowsA, colsA_rowsB)]);
			}
		}

		template<Options Option>
		static inline __m256 loadB(const float* b, const size_t innerIndex, const size_t currentColB, const size_t colsB, const size_t colsA_rowsB)
		{
			if constexpr (Option == Options::RowMajor) {
				return _mm256_set_ps(
					b[indexB<Option>(innerIndex + 7, currentColB, colsB, colsA_rowsB)],
					b[indexB<Option>(innerIndex + 6, currentColB, colsB, colsA_rowsB)],
					b[indexB<Option>(innerIndex + 5, currentColB, colsB, colsA_rowsB)],
					b[indexB<Option>(innerIndex + 4, currentColB, colsB, colsA_rowsB)],
					b[indexB<Option>(innerIndex + 3, currentColB, colsB, colsA_rowsB)],
					b[indexB<Option>(innerIndex + 2, currentColB, colsB, colsA_rowsB)],
					b[indexB<Option>(innerIndex + 1, currentColB, colsB, colsA_rowsB)],
					b[indexB<Option>(innerIndex, currentColB, colsB, colsA_rowsB)]);
			}
			else {
				return _mm256_loadu_ps(&b[indexB<Option>(innerIndex, currentColB, colsB, colsA_rowsB)]);
			}
		}

		//END: multiplication array utility functions

		template<Options Option>
		static void mul_avx2(float* result, const float* a, const float* b, size_t rowsA, size_t colsB, size_t colsA_rowsB)
		{
			mul_avx<Option>(result, a, b, rowsA, colsB, colsA_rowsB);
		}

		template<Options Option>
		static void mul_avx(float* result, const float* a, const float* b, size_t rowsA, size_t colsB, size_t colsA_rowsB)
		{
			for (size_t currentRowA = 0; currentRowA < rowsA; ++currentRowA)
			{
				for (size_t currentColB = 0; currentColB < colsB; ++currentColB)
				{
					__m256 sum = _mm256_setzero_ps();
					size_t innerIndex = 0;
					for (; innerIndex + 7 < colsA_rowsB; innerIndex += 8)
					{
						__m256 a_vec = loadA<Option>(a, currentRowA, innerIndex, rowsA, colsA_rowsB);
						__m256 b_vec = loadB<Option>(b, innerIndex, currentColB, colsB, colsA_rowsB);

						sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
					}
					float tmp[8];
					_mm256_storeu_ps(tmp, sum);
					float finalSum = 0.0f;
					for (int i = 0; i < 8; ++i) {
						finalSum += tmp[i];
					}
					for (; innerIndex < colsA_rowsB; ++innerIndex)
					{
						finalSum += a[indexA<Option>(currentRowA, innerIndex, rowsA, colsA_rowsB)] * b[indexB<Option>(innerIndex, currentColB, colsB, colsA_rowsB)];
					}
					result[indexResult<Option>(currentRowA, currentColB, rowsA, colsB)] = finalSum;
				}
			}
		}

		template<Options Option>
		static void mul_sse4_2(float* result, const float* a, const float* b, const size_t rowsA, const size_t colsB, const size_t colsA_rowsB)
		{
			mul_fallback<Option>(result, a, b, rowsA, colsB, colsA_rowsB);
		}

		template<Options Option>
		static void mul_sse4_1(float* result, const float* a, const float* b, const size_t rowsA, const size_t colsB, const size_t colsA_rowsB)
		{
			mul_fallback<Option>(result, a, b, rowsA, colsB, colsA_rowsB);
		}

		template<Options Option>
		static void mul_ssse3(float* result, const float* a, const float* b, const size_t rowsA, const size_t colsB, const size_t colsA_rowsB)
		{
			mul_fallback<Option>(result, a, b, rowsA, colsB, colsA_rowsB);
		}

		template<Options Option>
		static void mul_sse3(float* result, const float* a, const float* b, const size_t rowsA, const size_t colsB, const size_t colsA_rowsB)
		{
			mul_fallback<Option>(result, a, b, rowsA, colsB, colsA_rowsB);
		}


		template<Options Option>
		static void mul_fallback(float* result, const float* a, const float* b, const size_t rowsA, const size_t colsB, const size_t colsA_rowsB){
			for (size_t i = 0; i < rowsA; ++i) {
				for (size_t j = 0; j < colsB; ++j) {
					float sum = 0;
					for (size_t k = 0; k < colsA_rowsB; ++k) {
						sum += a[indexA<Option>(i, k, rowsA, colsA_rowsB)] * b[indexB<Option>(k, j, colsB, colsA_rowsB)];
					}
					result[indexResult<Option>(i, j, rowsA, colsB)] = sum;
				}
			}
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
			__m256 ymm0 = _mm256_set1_ps(scalar);
			const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
			size_t i = 0;

			for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				ymm1 = _mm256_mul_ps(ymm1, ymm0);
				_mm256_storeu_ps(a + i, ymm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] *= scalar;
			}
		}


		static void mul_scalar_sse4_2(float* a, float scalar, size_t size)
		{
			mul_scalar_sse3(a, scalar, size);
		}

		static void mul_scalar_sse4_1(float* a, float scalar, size_t size)
		{
			mul_scalar_sse3(a, scalar, size);
		}

		static void mul_scalar_ssse3(float* a, float scalar, size_t size)
		{
			mul_scalar_sse3(a, scalar, size);
		}

		static void mul_scalar_sse3(float* a, float scalar, size_t size)
		{
			__m128 xmm0 = _mm_set1_ps(scalar);
			const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
			size_t i = 0;

			for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				xmm1 = _mm_mul_ps(xmm1, xmm0);
				_mm_storeu_ps(a + i, xmm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] *= scalar;
			}
		}


		static void mul_scalar_fallback(float* a, float scalar, size_t size)
		{
			for (size_t i = 0; i < size; ++i) {
				a[i] *= scalar;
			}
		}

		//END: multiplication scalar
		//----------------------------------------------------------------------------

		//BEGIN: division scalar
		//----------------------------------------------------------------------------

		static void div_scalar_avx2(float* a, float scalar, size_t size)
		{
			div_scalar_avx(a, scalar, size);
		}

		static void div_scalar_avx(float* a, float scalar, size_t size)
		{
			__m256 ymm0 = _mm256_set1_ps(scalar);
			const size_t avx_limit = size - (size % AVX_SIMD_WIDTH);
			size_t i = 0;

			for (; i < avx_limit; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				ymm1 = _mm256_div_ps(ymm1, ymm0);
				_mm256_storeu_ps(a + i, ymm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] /= scalar;
			}
		}


		static void div_scalar_sse4_2(float* a, float scalar, size_t size)
		{
			div_scalar_sse3(a, scalar, size);
		}

		static void div_scalar_sse4_1(float* a, float scalar, size_t size)
		{
			div_scalar_sse3(a, scalar, size);
		}

		static void div_scalar_ssse3(float* a, float scalar, size_t size)
		{
			div_scalar_sse3(a, scalar, size);
		}

		static void div_scalar_sse3(float* a, float scalar, size_t size)
		{
			__m128 xmm0 = _mm_set1_ps(scalar);
			const size_t sse_limit = size - (size % SSE_SIMD_WIDTH);
			size_t i = 0;

			for (; i < sse_limit; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				xmm1 = _mm_div_ps(xmm1, xmm0);
				_mm_storeu_ps(a + i, xmm1);
			}

			// Handle any remainder
			for (; i < size; ++i) {
				a[i] /= scalar;
			}
		}


		static void div_scalar_fallback(float* a, float scalar, size_t size)
		{
			for (size_t i = 0; i < size; ++i) {
				a[i] /= scalar;
			}
		}

		//END: division scalar
		//----------------------------------------------------------------------------


	};

}