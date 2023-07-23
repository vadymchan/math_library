#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "All.h"

TEST(MatrixTest, ConstructorDestructorFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    // If the constructor and destructor work correctly, this test will pass
}

TEST(MatrixTest, ElementAccessFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    EXPECT_EQ(matrix.coeff(0, 1), 2);
    EXPECT_EQ(matrix.coeff(1, 0), 3);
    EXPECT_EQ(matrix.coeff(1, 1), 4);
}

TEST(MatrixTest, ConstructorDestructorFailureFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    // This test will pass because the matrix is not null after construction
    EXPECT_NE(&matrix, nullptr);
}

TEST(MatrixTest, ElementAccessFailureFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    // These tests will pass because the expected values match the actual values
    EXPECT_NE(matrix.coeff(0, 0), 2);
    EXPECT_NE(matrix.coeff(0, 1), 1);
    EXPECT_NE(matrix.coeff(1, 0), 4);
    EXPECT_NE(matrix.coeff(1, 1), 3);
}

TEST(MatrixTest, ElementModificationFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_EQ(matrix.coeff(0, 0), 2);
}

TEST(MatrixTest, ElementModificationFailureFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_NE(matrix.coeff(0, 0), 2);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_NE(matrix.coeff(0, 0), 1);
}

TEST(MatrixTest, OutOfBoundsAccessFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    EXPECT_DEATH(matrix.coeff(2, 2), "Index out of bounds");
    EXPECT_DEATH(matrix.coeffRef(2, 2), "Index out of bounds");
}

TEST(MatrixTest, OperatorAccessFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix(0, 0) = 1;
    EXPECT_EQ(matrix(0, 0), 1);
    matrix(0, 0) = 2;
    EXPECT_EQ(matrix(0, 0), 2);
}

TEST(MatrixTest, OperatorAccessConstFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix(0, 0))>>));
}

TEST(MatrixTest, CoeffAccessConstFloat) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix.coeff(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix.coeff(0, 0))>>));
}

TEST(MatrixTest, StackAllocationFloat) 
{
    // This matrix should be small enough to be allocated on the stack
    math::Matrix<float, 2, 2> matrix;
    EXPECT_FALSE(matrix.UseHeap);
}

TEST(MatrixTest, HeapAllocationFloat) 
{
    // This matrix should be large enough to be allocated on the heap
    math::Matrix<float, 100, 100> matrix;
    EXPECT_TRUE(matrix.UseHeap);
}

TEST(MatrixTest, AdditionFloat)
{
    constexpr int kRows = 2;
    constexpr int kColumns = 2;

    math::Matrix<float, kRows, kColumns> matrix1;
    math::Matrix<float, kRows, kColumns> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    math::Matrix<float, kRows, kColumns> matrix3 = matrix1 + matrix2;

    // Check that each element of the result is the sum of the corresponding elements of matrix1 and matrix2
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            EXPECT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) + matrix2.coeff(i, j));
        }
    }
}

TEST(MatrixTest, AdditionFailureFloat)
{
    math::Matrix<float, 4, 4, math::Options::COLUMN_MAJOR> matrix1;
    math::Matrix<float, 4, 4, math::Options::COLUMN_MAJOR> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    math::Matrix<float, 4, 4, math::Options::COLUMN_MAJOR> matrix3 = matrix1 + matrix2;

    // Check that each element of the result is not the sum of the corresponding elements of matrix1 and matrix2 plus 1
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NE(matrix3.coeff(i, j), matrix1.coeff(i, j) + matrix2.coeff(i, j) + 1);
        }
    }
}

TEST(MatrixTest, ScalarAdditionFloat)
{
    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix2 = matrix1 + 10;

    // Check that each element of the result is the corresponding element of matrix1 plus 10
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) + 10);
        }
    }
}

TEST(MatrixTest, ScalarAdditionFailureFloat)
{
    math::Matrix<float, 4, 4> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4> matrix2 = matrix1 + 10;

    // Check that each element of the result is not the corresponding element of matrix1 plus 11
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) + 11);
        }
    }
}

TEST(MatrixTest, SubtractionFloat)
{
    constexpr int kRows = 2;
    constexpr int kColumns = 2;

    math::Matrix<int, kRows, kColumns> matrix1;
    math::Matrix<int, kRows, kColumns> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    math::Matrix<int, kRows, kColumns> matrix3 = matrix1 - matrix2;

    // Check that each element of the result is the difference of the corresponding elements of matrix1 and matrix2
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            EXPECT_EQ(matrix3.coeff(i, j), matrix1.coeff(i, j) - matrix2.coeff(i, j));
        }
    }
}

TEST(MatrixTest, ScalarSubtractionFloat)
{
    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix2 = matrix1 - 10;

    // Check that each element of the result is the corresponding element of matrix1 minus 10
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) - 10);
        }
    }
}


TEST(MatrixTest, MultiplicationRowMajorFloat)
{
    constexpr int kRows = 2;
    constexpr int kColumns = 2;

    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix1;
    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    // Print initial matrices
    std::cout << "Matrix 1:\n" << matrix1 << "\nMatrix 2:\n" << matrix2 << std::endl;

    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix3 = matrix1 * matrix2;

    // Print result matrix
    std::cout << "Result Matrix:\n" << matrix3 << std::endl;

    // Check that each element of the result is the correct multiplication of the corresponding rows and columns of matrix1 and matrix2
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            float expected_value = 0;
            for (int k = 0; k < kColumns; ++k) {
                expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
            }
            EXPECT_EQ(matrix3.coeff(i, j), expected_value);
        }
    }
}

TEST(MatrixTest, MultiplicationColumnMajorFloat)
{
    constexpr int kRows = 2;
    constexpr int kColumns = 2;

    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix1;
    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix2;

    // Populate matrix1 and matrix2 with some values
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
        }
    }

    // Print initial matrices
    std::cout << "Matrix 1:\n" << matrix1 << "\nMatrix 2:\n" << matrix2 << std::endl;

    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix3 = matrix1 * matrix2;

    // Print result matrix
    std::cout << "Result Matrix:\n" << matrix3 << std::endl;

    // Check that each element of the result is the correct multiplication of the corresponding rows and columns of matrix1 and matrix2
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kColumns; ++j) {
            float expected_value = 0;
            for (int k = 0; k < kColumns; ++k) {
                expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
            }
            EXPECT_EQ(matrix3.coeff(i, j), expected_value);
        }
    }
}

//TEST(MatrixTest, MultiplicationRowMajorFloat)
//{
//    constexpr int kRows = 2;
//    constexpr int kColumns = 2;
//
//    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix1;
//    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix2;
//
//    // Populate matrix1 and matrix2 with some values
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            matrix1.coeffRef(i, j) = i * 4 + j + 1;
//            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
//        }
//    }
//
//    math::Matrix<float, kRows, kColumns, math::Options::ROW_MAJOR> matrix3 = matrix1 * matrix2;
//
//    // Check that each element of the result is the correct multiplication of the corresponding rows and columns of matrix1 and matrix2
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            float expected_value = 0;
//            for (int k = 0; k < kColumns; ++k) {
//                expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
//            }
//            EXPECT_EQ(matrix3.coeff(i, j), expected_value);
//        }
//    }
//}
//
//TEST(MatrixTest, MultiplicationColumnMajorFloat)
//{
//    constexpr int kRows = 2;
//    constexpr int kColumns = 2;
//
//    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix1;
//    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix2;
//
//    // Populate matrix1 and matrix2 with some values
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            matrix1.coeffRef(i, j) = i * 4 + j + 1;
//            matrix2.coeffRef(i, j) = (i * 4 + j + 1) * 2;
//        }
//    }
//
//    math::Matrix<float, kRows, kColumns, math::Options::COLUMN_MAJOR> matrix3 = matrix1 * matrix2;
//
//    // Check that each element of the result is the correct multiplication of the corresponding rows and columns of matrix1 and matrix2
//    for (int i = 0; i < kRows; ++i) {
//        for (int j = 0; j < kColumns; ++j) {
//            float expected_value = 0;
//            for (int k = 0; k < kColumns; ++k) {
//                expected_value += matrix1.coeff(i, k) * matrix2.coeff(k, j);
//            }
//            EXPECT_EQ(matrix3.coeff(i, j), expected_value);
//        }
//    }
//}


TEST(MatrixTest, ScalarDivisionFloat)
{
    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4, math::Options::ROW_MAJOR> matrix2 = matrix1 / 10;

    // Check that each element of the result is the corresponding element of matrix1 divided by 10
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQ(matrix2.coeff(i, j), matrix1.coeff(i, j) / 10);
        }
    }
}

TEST(MatrixTest, ScalarDivisionFailureFloat)
{
    math::Matrix<float, 4, 4> matrix1;

    // Populate matrix1 with some values
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix1.coeffRef(i, j) = i * 4 + j + 1;
        }
    }

    math::Matrix<float, 4, 4> matrix2 = matrix1 / 10;

    // Check that each element of the result is not the corresponding element of matrix1 divided by 11
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NE(matrix2.coeff(i, j), matrix1.coeff(i, j) / 11);
        }
    }
}

//========================================= BENCHMARKING =========================================
static void BM_MatrixCreationStack(benchmark::State& state) 
{
    for (auto _ : state) 
{
        math::Matrix<float, 2, 2> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_MatrixCreationStack);

static void BM_MatrixCreationHeap(benchmark::State& state) 
{
    for (auto _ : state) 
{
        math::Matrix<float, 100, 100> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_MatrixCreationHeap);

static void BM_MatrixElementAccess(benchmark::State& state) 
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state) 
{
        auto val = matrix.coeff(50, 50);
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_MatrixElementAccess);

static void BM_MatrixElementAccessRef(benchmark::State& state) 
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state) 
{
        matrix.coeffRef(50, 50) = 1;
    }
}
BENCHMARK(BM_MatrixElementAccessRef);

static void BM_MatrixOperatorAccess(benchmark::State& state) 
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state) 
{
        auto val = matrix(50, 50);
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_MatrixOperatorAccess);

static void BM_MatrixOperatorAccessRef(benchmark::State& state) 
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state) 
{
        matrix(50, 50) = 1;
    }
}
BENCHMARK(BM_MatrixOperatorAccessRef);


//BEGIN: addition benchmark
//---------------------------------------------------------------------------

static void BM_MatrixAddition(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix1;
    math::Matrix<float, 100, 100> matrix2;
    for (auto _ : state)
    {
        auto matrix3 = matrix1 + matrix2;
        benchmark::DoNotOptimize(matrix3);
    }
}
BENCHMARK(BM_MatrixAddition);

static void BM_MatrixAdditionInPlace(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix1;
    math::Matrix<float, 100, 100> matrix2;
    for (auto _ : state)
    {
        matrix1 += matrix2;
    }
}
BENCHMARK(BM_MatrixAdditionInPlace);

static void BM_MatrixScalarAddition(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        auto matrix2 = matrix + 1.0f;
        benchmark::DoNotOptimize(matrix2);
    }
}
BENCHMARK(BM_MatrixScalarAddition);

static void BM_MatrixScalarAdditionInPlace(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix += 1.0f;
    }
}
BENCHMARK(BM_MatrixScalarAdditionInPlace);


//END: addition benchmark
//---------------------------------------------------------------------------

//BEGIN: subtraction benchmark
//---------------------------------------------------------------------------

static void BM_MatrixSubtraction(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix1;
    math::Matrix<float, 100, 100> matrix2;
    for (auto _ : state)
    {
        auto matrix3 = matrix1 - matrix2;
        benchmark::DoNotOptimize(matrix3);
    }
}
BENCHMARK(BM_MatrixSubtraction);

static void BM_MatrixSubtractionInPlace(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix1;
    math::Matrix<float, 100, 100> matrix2;
    for (auto _ : state)
    {
        matrix1 -= matrix2;
    }
}
BENCHMARK(BM_MatrixSubtractionInPlace);

static void BM_MatrixScalarSubtraction(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        auto matrix2 = matrix - 1.0f;
        benchmark::DoNotOptimize(matrix2);
    }
}
BENCHMARK(BM_MatrixScalarSubtraction);

static void BM_MatrixScalarSubtractionInPlace(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix -= 1.0f;
    }
}
BENCHMARK(BM_MatrixScalarSubtractionInPlace);

//END: subtraction benchmark
//---------------------------------------------------------------------------

//BEGIN: division benchmark
//---------------------------------------------------------------------------

static void BM_MatrixScalarDivision(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        auto matrix2 = matrix / 1.0f;
        benchmark::DoNotOptimize(matrix2);
    }
}
BENCHMARK(BM_MatrixScalarDivision);

static void BM_MatrixScalarDivisionInPlace(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix /= 1.0f;
    }
}
BENCHMARK(BM_MatrixScalarDivisionInPlace);

//END: division benchmark
//---------------------------------------------------------------------------

//BEGIN: multiplication benchmark
//---------------------------------------------------------------------------

static void BM_MatrixMultiplication(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix1;
    math::Matrix<float, 100, 100> matrix2;
    for (auto _ : state)
    {
        auto matrix3 = matrix1 * matrix2;
        benchmark::DoNotOptimize(matrix3);
    }
}
BENCHMARK(BM_MatrixMultiplication);

static void BM_MatrixMultiplicationInPlace(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix1;
    math::Matrix<float, 100, 100> matrix2;
    for (auto _ : state)
    {
        matrix1 *= matrix2;
    }
}
BENCHMARK(BM_MatrixMultiplicationInPlace);

static void BM_MatrixScalarMultiplication(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        auto matrix2 = matrix * 2.0f;
        benchmark::DoNotOptimize(matrix2);
    }
}
BENCHMARK(BM_MatrixScalarMultiplication);

static void BM_MatrixScalarMultiplicationInPlace(benchmark::State& state)
{
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix *= 2.0f;
    }
}
BENCHMARK(BM_MatrixScalarMultiplicationInPlace);

//END: multiplication benchmark
//---------------------------------------------------------------------------


//========================================= END::BENCHMARKING =========================================

//END: division benchmark
//---------------------------------------------------------------------------

//int main(int argc, char** argv) 
//{
//    ::testing::InitGoogleTest(&argc, argv);
//    int test_result = RUN_ALL_TESTS();
//    if (test_result != 0) 
//{
//        return test_result;
//    }
//
//    ::benchmark::Initialize(&argc, argv);
//    ::benchmark::RunSpecifiedBenchmarks();
//    return 0;
//}
//    return 0;

//}
//}







int main(int argc, char** argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    int test_result = RUN_ALL_TESTS();
    if (test_result != 0) 
{
        return test_result;
    }

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}

//============================================================================

//#define COLUMN_MAJOR_TRANSPOSED


void mul_avx_row_major(float* result, const float* a, const float* b, size_t size) {
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

#ifdef COLUMN_MAJOR_TRANSPOSED



void mul_avx_col_major(float* result, const float* a, const float* b, size_t size) {
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
            result[i * size + j] = _mm_cvtss_f32(vlow) + tail_sum;
        }
    }
}

#else


void mul_avx_col_major(float* result, const float* a, const float* b, size_t size) {
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



#endif // COLUMN_MAJOR_TRANSPOSED

void transpose(float* a, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i + 1; j < size; ++j) {
            std::swap(a[i * size + j], a[j * size + i]);
        }
    }
}

//int main() {
//    const size_t size = 11;
//    float a[size * size];
//    for (size_t i = 0; i < size * size; ++i) {
//        a[i] = i + 1;
//    }
//
//    float result[size * size];
//    float expected[size * size];
//
//    // Calculate expected result
//    for (size_t i = 0; i < size; ++i) {
//        for (size_t j = 0; j < size; ++j) {
//            expected[i * size + j] = 0;
//            for (size_t k = 0; k < size; ++k) {
//                expected[i * size + j] += a[i * size + k] * a[k * size + j];
//            }
//        }
//    }
//
//    mul_avx_row_major(result, a, a, size);
//
//    std::cout << "Row-major result:" << std::endl;
//    for (size_t i = 0; i < size; ++i) {
//        for (size_t j = 0; j < size; ++j) {
//            std::cout << result[i * size + j] << ' ';
//        }
//        std::cout << std::endl;
//    }
//
//    // Expected output:
//    std::cout << "Expected result:" << std::endl;
//    for (size_t i = 0; i < size; ++i) {
//        for (size_t j = 0; j < size; ++j) {
//            std::cout << expected[i * size + j] << ' ';
//        }
//        std::cout << std::endl;
//    }
//
//    // Transpose the matrix
//    transpose(a, size);
//
//    mul_avx_col_major(result, a, a, size);
//
//    std::cout << "Column-major result:" << std::endl;
//    for (size_t i = 0; i < size; ++i) {
//        for (size_t j = 0; j < size; ++j) {
//            std::cout << result[i * size + j] << ' ';
//        }
//        std::cout << std::endl;
//    }
//
//    return 0;
#include <iostream>
using namespace std;

void transpose_avx(float* src, float* dst, const int n, const int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            dst[j * n + i] = src[i * m + j];
        }
    }
}



void printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int n = 2;
    int m = 3;
    float* src = new float[n * m] {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  // This is a 3x2 matrix

    cout << "Original matrix:" << endl;
    printMatrix(src, n, m);

    float* dst = new float[n * m];
    transpose_avx(src, dst, n, m);

    cout << "Transposed matrix:" << endl;
    printMatrix(dst, m, n);  // The transposed matrix is 2x3

    delete[] src;
    delete[] dst;

    return 0;
}
