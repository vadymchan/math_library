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
		using MulFunc = void(*)(float*, const float*, const float*, size_t);

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

			

			// Handling remaining elements
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

			size_t aligned_size = (size / SSE_SIMD_WIDTH) * SSE_SIMD_WIDTH;
			size_t i = 0;

			for (i = 0; i < aligned_size; i += SSE_SIMD_WIDTH) {
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
			size_t i = 0;

			for (; i < size; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				xmm1 = _mm_add_ps(xmm1, xmm0);
				_mm_storeu_ps(a + i, xmm1);
			}

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
			size_t aligned_size = (size / AVX_SIMD_WIDTH) * AVX_SIMD_WIDTH;
			size_t i = 0;

			for (i = 0; i < aligned_size; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				__m256 ymm2 = _mm256_loadu_ps(b + i);
				ymm1 = _mm256_sub_ps(ymm1, ymm2);
				_mm256_storeu_ps(a + i, ymm1);
			}

			// Handling remaining elements
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
			size_t aligned_size = (size / SSE_SIMD_WIDTH) * SSE_SIMD_WIDTH;
			size_t i = 0;

			for (i = 0; i < aligned_size; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				__m128 xmm2 = _mm_loadu_ps(b + i);
				xmm1 = _mm_sub_ps(xmm1, xmm2);
				_mm_storeu_ps(a + i, xmm1);
			}

			// Handling remaining elements
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
			size_t i = 0;

			for (; i < size; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				ymm1 = _mm256_sub_ps(ymm1, ymm0);
				_mm256_storeu_ps(a + i, ymm1);
			}

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
			size_t i = 0;

			for (; i < size; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				xmm1 = _mm_sub_ps(xmm1, xmm0);
				_mm_storeu_ps(a + i, xmm1);
			}

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

		template<Options Option>
		static void mul_avx2(float* result, const float* a, const float* b, size_t size)
		{
			mul_avx<Option>(result, a, b, size);
		}

		template<math::Options Option>
		static void mul_avx(float* result, const float* a, const float* b, size_t size)
		{
			if constexpr (Option == math::Options::ROW_MAJOR) {
				for (size_t i = 0; i < size; ++i) {
					for (size_t j = 0; j < size; ++j) {
						__m256 sum = _mm256_setzero_ps();
						size_t k = 0;
						for (; k + 7 < size; k += 8) {
							__m256 a_vec = _mm256_loadu_ps(&a[i * size + k]);
							__m256 b_vec = _mm256_set_ps(b[(k + 7) * size + j], b[(k + 6) * size + j], b[(k + 5) * size + j], b[(k + 4) * size + j],
								b[(k + 3) * size + j], b[(k + 2) * size + j], b[(k + 1) * size + j], b[k * size + j]);

							sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
						}
						sum = _mm256_hadd_ps(sum, sum);
						sum = _mm256_hadd_ps(sum, sum);
						__m128 vlow = _mm256_castps256_ps128(sum);
						__m128 vhigh = _mm256_extractf128_ps(sum, 1);
						vlow = _mm_add_ps(vlow, vhigh);
						float tail_sum = 0;
						for (; k < size; ++k) {
							tail_sum += a[i * size + k] * b[k * size + j];
						}
						result[i * size + j] = _mm_cvtss_f32(vlow) + tail_sum;
					}
				}
			}
			else if constexpr (Option == math::Options::COLUMN_MAJOR) {
				for (size_t i = 0; i < size; ++i) {
					for (size_t j = 0; j < size; ++j) {
						__m256 sum = _mm256_setzero_ps();
						size_t k = 0;
						for (; k + 7 < size; k += 8) {
							__m256 a_vec = _mm256_set_ps(b[(k + 7) * size + i], b[(k + 6) * size + i], b[(k + 5) * size + i], b[(k + 4) * size + i],
								b[(k + 3) * size + i], b[(k + 2) * size + i], b[(k + 1) * size + i], b[k * size + i]);
							__m256 b_vec = _mm256_loadu_ps(&a[j * size + k]);
							sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
						}
						sum = _mm256_hadd_ps(sum, sum);
						sum = _mm256_hadd_ps(sum, sum);
						__m128 vlow = _mm256_castps256_ps128(sum);
						__m128 vhigh = _mm256_extractf128_ps(sum, 1);
						vlow = _mm_add_ps(vlow, vhigh);
						float tail_sum = 0;
						for (; k < size; ++k) {
							tail_sum += a[k * size + i] * b[j * size + k];
						}
						result[j * size + i] = _mm_cvtss_f32(vlow) + tail_sum;
					}
				}
			}
		}


		template<Options Option>
		static void mul_sse4_2(float* result, const float* a, const float* b, size_t size)
		{
			mul_fallback<Option>(result, a, b, size);
		}

		template<Options Option>
		static void mul_sse4_1(float* result, const float* a, const float* b, size_t size)
		{
			mul_fallback<Option>(result, a, b, size);
		}

		template<Options Option>
		static void mul_ssse3(float* result, const float* a, const float* b, size_t size)
		{
			mul_fallback<Option>(result, a, b, size);
		}

		template<Options Option>
		static void mul_sse3(float* result, const float* a, const float* b, size_t size)
		{
			mul_fallback<Option>(result, a, b, size);
		}

		template<Options Option>
		static void mul_fallback(float* result, const float* a, const float* b, size_t size, size_t dim)
		{
			if constexpr (Option == Options::COLUMN_MAJOR) {
				for (size_t i = 0; i < dim; ++i) {
					for (size_t j = 0; j < dim; ++j) {
						float sum = 0;
						for (size_t k = 0; k < dim; ++k) {
							sum += a[i + k * dim] * b[k + j * dim];
						}
						result[i + j * dim] = sum;
					}
				}
			}
			else if constexpr (Option == Options::ROW_MAJOR) {
				for (size_t i = 0; i < dim; ++i) {
					for (size_t j = 0; j < dim; ++j) {
						float sum = 0;
						for (size_t k = 0; k < dim; ++k) {
							sum += a[i * dim + k] * b[k * dim + j];
						}
						result[i * dim + j] = sum;
					}
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
			size_t i = 0;

			for (; i < size; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				ymm1 = _mm256_mul_ps(ymm1, ymm0);
				_mm256_storeu_ps(a + i, ymm1);
			}

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
			size_t i = 0;

			for (; i < size; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				xmm1 = _mm_mul_ps(xmm1, xmm0);
				_mm_storeu_ps(a + i, xmm1);
			}

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
			size_t i = 0;

			for (; i < size; i += AVX_SIMD_WIDTH) {
				__m256 ymm1 = _mm256_loadu_ps(a + i);
				ymm1 = _mm256_div_ps(ymm1, ymm0);
				_mm256_storeu_ps(a + i, ymm1);
			}

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
			size_t i = 0;

			for (; i < size; i += SSE_SIMD_WIDTH) {
				__m128 xmm1 = _mm_loadu_ps(a + i);
				xmm1 = _mm_div_ps(xmm1, xmm0);
				_mm_storeu_ps(a + i, xmm1);
			}

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