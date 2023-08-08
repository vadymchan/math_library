#pragma once

#include <benchmark/benchmark.h>

#include <All.h>


static void BM_MatrixCreationStack(benchmark::State& state);
static void BM_MatrixCreationHeap(benchmark::State& state);
static void BM_MatrixElementAccess(benchmark::State& state);
static void BM_MatrixElementAccessRef(benchmark::State& state);
static void BM_MatrixOperatorAccess(benchmark::State& state);
static void BM_MatrixOperatorAccessRef(benchmark::State& state);
static void BM_MatrixAddition(benchmark::State& state);
static void BM_MatrixAdditionInPlace(benchmark::State& state);
static void BM_MatrixScalarAddition(benchmark::State& state);
static void BM_MatrixScalarAdditionInPlace(benchmark::State& state);
static void BM_MatrixSubtraction(benchmark::State& state);
static void BM_MatrixSubtractionInPlace(benchmark::State& state);
static void BM_MatrixScalarSubtraction(benchmark::State& state);
static void BM_MatrixScalarSubtractionInPlace(benchmark::State& state);
static void BM_MatrixMultiplication(benchmark::State& state);
static void BM_MatrixMultiplicationInPlace(benchmark::State& state);
static void BM_MatrixScalarMultiplication(benchmark::State& state);
static void BM_MatrixScalarMultiplicationInPlace(benchmark::State& state);
static void BM_MatrixScalarDivision(benchmark::State& state);
static void BM_MatrixScalarDivisionInPlace(benchmark::State& state);
static void BM_MatrixTrace(benchmark::State& state);

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

static void BM_MatrixDeterminant(benchmark::State& state) {
    for (auto _ : state) {
        math::Matrix<float, 5, 5> m;
        m.determinant();
    }
}
BENCHMARK(BM_MatrixDeterminant);

static void BM_MatrixInverse(benchmark::State& state) {
    for (auto _ : state) {
        math::Matrix<float, 5, 5> m;  
        m.inverse();            
    }
}
BENCHMARK(BM_MatrixInverse);

static void BM_MatrixRank(benchmark::State& state) {
    for (auto _ : state) {
        math::Matrix<float, 5, 5> m;  
        m.rank();               
    }
}
BENCHMARK(BM_MatrixRank);

static void BM_MatrixMagnitude(benchmark::State& state) {
    for (auto _ : state) {
        math::Matrix<float, 5, 1> m; 
        m.magnitude();         
    }
}
BENCHMARK(BM_MatrixMagnitude);

static void BM_MatrixNormalize(benchmark::State& state) {
    for (auto _ : state) {
        math::Matrix<float, 5, 1> m;  
        m.normalize();          
    }
}
BENCHMARK(BM_MatrixNormalize);

//BEGIN: trace benchmark
//---------------------------------------------------------------------------

static void BM_MatrixTrace(benchmark::State& state)
{
    // Create a 100x100 matrix for the benchmark
    math::Matrix<float, 100, 100> matrix;
    for (auto _ : state)
    {
        float trace = matrix.trace();
        benchmark::DoNotOptimize(trace);
    }
}
BENCHMARK(BM_MatrixTrace);

//END: trace benchmark
//---------------------------------------------------------------------------



//========================================= END::BENCHMARKING =========================================

