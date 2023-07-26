#pragma once

#include <benchmark/benchmark.h>

#include <All.h>


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

//========================================= BEGIN::INT_BENCHMARKING =========================================

static void BM_MatrixCreationStackInt(benchmark::State& state)
{
    for (auto _ : state)
    {
        math::Matrix<int, 2, 2> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_MatrixCreationStackInt);

static void BM_MatrixCreationHeapInt(benchmark::State& state)
{
    for (auto _ : state)
    {
        math::Matrix<int, 100, 100> matrix;
        benchmark::DoNotOptimize(matrix);
    }
}
BENCHMARK(BM_MatrixCreationHeapInt);

static void BM_MatrixElementAccessInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        auto val = matrix.coeff(50, 50);
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_MatrixElementAccessInt);

static void BM_MatrixElementAccessRefInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix.coeffRef(50, 50) = 1;
    }
}
BENCHMARK(BM_MatrixElementAccessRefInt);

static void BM_MatrixOperatorAccessInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        auto val = matrix(50, 50);
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_MatrixOperatorAccessInt);

static void BM_MatrixOperatorAccessRefInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix(50, 50) = 1;
    }
}
BENCHMARK(BM_MatrixOperatorAccessRefInt);

//BEGIN: addition benchmark
//---------------------------------------------------------------------------

static void BM_MatrixAdditionInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix1;
    math::Matrix<int, 100, 100> matrix2;
    for (auto _ : state)
    {
        auto matrix3 = matrix1 + matrix2;
        benchmark::DoNotOptimize(matrix3);
    }
}
BENCHMARK(BM_MatrixAdditionInt);

static void BM_MatrixAdditionInPlaceInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix1;
    math::Matrix<int, 100, 100> matrix2;
    for (auto _ : state)
    {
        matrix1 += matrix2;
    }
}
BENCHMARK(BM_MatrixAdditionInPlaceInt);

static void BM_MatrixScalarAdditionInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        auto matrix2 = matrix + 1;
        benchmark::DoNotOptimize(matrix2);
    }
}
BENCHMARK(BM_MatrixScalarAdditionInt);

static void BM_MatrixScalarAdditionInPlaceInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix += 1;
    }
}
BENCHMARK(BM_MatrixScalarAdditionInPlaceInt);

//END: addition benchmark
//---------------------------------------------------------------------------

//BEGIN: subtraction benchmark
//---------------------------------------------------------------------------

static void BM_MatrixSubtractionInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix1;
    math::Matrix<int, 100, 100> matrix2;
    for (auto _ : state)
    {
        auto matrix3 = matrix1 - matrix2;
        benchmark::DoNotOptimize(matrix3);
    }
}
BENCHMARK(BM_MatrixSubtractionInt);

static void BM_MatrixSubtractionInPlaceInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix1;
    math::Matrix<int, 100, 100> matrix2;
    for (auto _ : state)
    {
        matrix1 -= matrix2;
    }
}
BENCHMARK(BM_MatrixSubtractionInPlaceInt);

static void BM_MatrixScalarSubtractionInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        auto matrix2 = matrix - 1;
        benchmark::DoNotOptimize(matrix2);
    }
}
BENCHMARK(BM_MatrixScalarSubtractionInt);

static void BM_MatrixScalarSubtractionInPlaceInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix -= 1;
    }
}
BENCHMARK(BM_MatrixScalarSubtractionInPlaceInt);

//END: subtraction benchmark
//---------------------------------------------------------------------------

//BEGIN: division benchmark
//---------------------------------------------------------------------------

static void BM_MatrixScalarDivisionInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        auto matrix2 = matrix / 1;
        benchmark::DoNotOptimize(matrix2);
    }
}
BENCHMARK(BM_MatrixScalarDivisionInt);

static void BM_MatrixScalarDivisionInPlaceInt(benchmark::State& state)
{
    math::Matrix<int, 100, 100> matrix;
    for (auto _ : state)
    {
        matrix /= 1;
    }
}
BENCHMARK(BM_MatrixScalarDivisionInPlaceInt);

//END: division benchmark
//---------------------------------------------------------------------------


//========================================= END::INT_BENCHMARKING =========================================

