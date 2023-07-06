#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
#include "All.h"

TEST(MatrixTest, ConstructorDestructor) {
    math::Matrix<int, 2, 2> matrix;
    // If the constructor and destructor work correctly, this test will pass
}

TEST(MatrixTest, ElementAccess) {
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    EXPECT_EQ(matrix.coeff(0, 1), 2);
    EXPECT_EQ(matrix.coeff(1, 0), 3);
    EXPECT_EQ(matrix.coeff(1, 1), 4);
}

TEST(MatrixTest, ConstructorDestructorFailure) {
    math::Matrix<int, 2, 2> matrix;
    // This test will pass because the matrix is not null after construction
    EXPECT_NE(&matrix, nullptr);
}

TEST(MatrixTest, ElementAccessFailure) {
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    // These tests will pass because the expected values match the actual values
    EXPECT_NE(matrix.coeff(0, 0), 2);
    EXPECT_NE(matrix.coeff(0, 1), 1);
    EXPECT_NE(matrix.coeff(1, 0), 4);
    EXPECT_NE(matrix.coeff(1, 1), 3);
}

TEST(MatrixTest, ElementModification) {
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_EQ(matrix.coeff(0, 0), 1);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_EQ(matrix.coeff(0, 0), 2);
}

TEST(MatrixTest, ElementModificationFailure) {
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    EXPECT_NE(matrix.coeff(0, 0), 2);
    matrix.coeffRef(0, 0) = 2;
    EXPECT_NE(matrix.coeff(0, 0), 1);
}

TEST(MatrixTest, OutOfBoundsAccess) {
    math::Matrix<int, 2, 2> matrix;
    EXPECT_DEATH(matrix.coeff(2, 2), "Index out of bounds");
    EXPECT_DEATH(matrix.coeffRef(2, 2), "Index out of bounds");
}

TEST(MatrixTest, OperatorAccess) {
    math::Matrix<int, 2, 2> matrix;
    matrix(0, 0) = 1;
    EXPECT_EQ(matrix(0, 0), 1);
    matrix(0, 0) = 2;
    EXPECT_EQ(matrix(0, 0), 2);
}

TEST(MatrixTest, OperatorAccessConst) {
    math::Matrix<int, 2, 2> matrix;
    matrix(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix(0, 0))>>));
}

TEST(MatrixTest, CoeffAccessConst) {
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1;
    const auto& const_matrix = matrix;
    EXPECT_EQ(const_matrix.coeff(0, 0), 1);
    // Check that the type of the returned reference is const
    EXPECT_TRUE((std::is_const_v<std::remove_reference_t<decltype(const_matrix.coeff(0, 0))>>));
}

static void BM_MatrixElementAccess(benchmark::State& state) {
    math::Matrix<int, 2, 2> matrix;
    matrix.coeffRef(0, 0) = 1; matrix.coeffRef(0, 1) = 2;
    matrix.coeffRef(1, 0) = 3; matrix.coeffRef(1, 1) = 4;
    for (auto _ : state) {
        auto val = matrix.coeff(1, 1);
        benchmark::DoNotOptimize(val);
    }
}

BENCHMARK(BM_MatrixElementAccess);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int test_result = RUN_ALL_TESTS();
    if (test_result != 0) {
        return test_result;
    }

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
