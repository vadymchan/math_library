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


//-------------------------------------------------------------

#include <benchmark/benchmark.h>

// The function to be benchmarked
static void MyBenchmarkFunction(benchmark::State& state) {
    // Perform setup here
    for (auto _ : state) {
        // Code to be benchmarked
    }
}

// Register the benchmark
BENCHMARK(MyBenchmarkFunction);

// Optionally, define custom arguments for the benchmark
// BENCHMARK(MyBenchmarkFunction)->Arg(10)->Arg(100)->Arg(1000);

// Define the main function
int main(int argc, char** argv) {
    // Initialize the benchmark
    benchmark::Initialize(&argc, argv);

    // Run the benchmark
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}


//-------------------------------------------------------------
//
//#include <gtest/gtest.h>
//
//// Example class to test
//class MyClass {
//public:
//    int Add(int a, int b) {
//        return a + b;
//    }
//};
//
//// Test fixture
//class MyTestClass : public testing::Test {
//protected:
//    MyClass myClass;
//};
//
//// Test case
//TEST_F(MyTestClass, AddTest) {
//    int result = myClass.Add(2, 3);
//    EXPECT_EQ(result, 5);
//}
//
//// Main function
//int main(int argc, char** argv) {
//    // Initialize Google Test
//    ::testing::InitGoogleTest(&argc, argv);
//
//    // Run all tests
//    return RUN_ALL_TESTS();
//}
