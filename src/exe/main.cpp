#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "All.h"

TEST(MatrixTest, ConstructorDestructor) 
{
    math::Matrix<float, 2, 2> matrix;
    // If the constructor and destructor work correctly, this test will pass
}

TEST(MatrixTest, ElementAccess) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    EXPECT_EQ(matrix.coeff(0, 1), 2);
    EXPECT_EQ(matrix.coeff(1, 0), 3);
    EXPECT_EQ(matrix.coeff(1, 1), 4);
}

TEST(MatrixTest, ConstructorDestructorFailure) 
{
    math::Matrix<float, 2, 2> matrix;
    // This test will pass because the matrix is not null after construction
    EXPECT_NE(&matrix, nullptr);
}

TEST(MatrixTest, ElementAccessFailure) 
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

TEST(MatrixTest, ElementModification) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_EQ(matrix.coeff(0, 0), 2);
}

TEST(MatrixTest, ElementModificationFailure) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_NE(matrix.coeff(0, 0), 2);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_NE(matrix.coeff(0, 0), 1);
}

TEST(MatrixTest, OutOfBoundsAccess) 
{
    math::Matrix<float, 2, 2> matrix;
    EXPECT_DEATH(matrix.coeff(2, 2), "Index out of bounds");
    EXPECT_DEATH(matrix.coeffRef(2, 2), "Index out of bounds");
}

TEST(MatrixTest, OperatorAccess) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix(0, 0) = 1;
    EXPECT_EQ(matrix(0, 0), 1);
    matrix(0, 0) = 2;
    EXPECT_EQ(matrix(0, 0), 2);
}

TEST(MatrixTest, OperatorAccessConst) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix(0, 0))>>));
}

TEST(MatrixTest, CoeffAccessConst) 
{
    math::Matrix<float, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix.coeff(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix.coeff(0, 0))>>));
}

TEST(MatrixTest, StackAllocation) 
{
    // This matrix should be small enough to be allocated on the stack
    math::Matrix<float, 2, 2> matrix;
    EXPECT_FALSE(matrix.UseHeap);
}

TEST(MatrixTest, HeapAllocation) 
{
    // This matrix should be large enough to be allocated on the heap
    math::Matrix<float, 100, 100> matrix;
    EXPECT_TRUE(matrix.UseHeap);
}

TEST(MatrixTest, Addition)
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

TEST(MatrixTest, AdditionFailure)
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

TEST(MatrixTest, ScalarAddition)
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

TEST(MatrixTest, ScalarAdditionFailure)
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
