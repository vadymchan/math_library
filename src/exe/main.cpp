//#include <vec3.h>
//
//#include <immintrin.h>
//
//#include <iostream>
//
//int main()
//{
//	vec3 v1(1, 2, 3);
//	vec3 v2(4, 5, 6);
//	vec3 v3 = v1 + v2;
//
//	std::cout << v3.x << ", " << v3.y << ", " << v3.z << std::endl;
//
//	return 0;
//}


#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

// Function to test
int add(int a, int b) {
    return a + b;
}

// Google Test
TEST(AddTest, HandlesPositiveInput) {
    ASSERT_EQ(6, add(2, 4));
}

// Google Benchmark
static void BM_Add(benchmark::State& state) {
    for (auto _ : state)
        add(2, 4);
}
// Register the function as a benchmark
BENCHMARK(BM_Add);

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::benchmark::Initialize(&argc, argv);
    // Run the tests
    int test_result = RUN_ALL_TESTS();
    // If tests are successful, run the benchmarks
    if (test_result == 0) {
        ::benchmark::RunSpecifiedBenchmarks();
    }
    return test_result;
}
