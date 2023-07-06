#include <fstream>
#include <iostream>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

int main() {
    
    std::string defines_file_path = "include/SIMDdefines.h";
    std::ofstream defines(defines_file_path);

    if (!defines) {
        std::cerr << "Failed to open " << defines_file_path << " for writing\n";
        return 1;
    } else {
        std::cout << "Successfully opened " << defines_file_path << " for writing\n";
    }
    //std::ofstream defines("../include/SIMDdefines.h");

    if (!defines) {
        std::cerr << "Failed to open defines.h for writing\n";
        return 1;
    }

#ifdef _MSC_VER
    int cpuInfo[4];

    // Get the highest function supported by CPUID
    __cpuid(cpuInfo, 0);

    if (cpuInfo[0] >= 1) {
        // Get the feature information
        __cpuid(cpuInfo, 1);

        if (cpuInfo[2] & (1 << 0)) {
            defines << "#define SUPPORTS_SSE3\n";
        }
        if (cpuInfo[2] & (1 << 9)) {
            defines << "#define SUPPORTS_SSSE3\n";
        }
        if (cpuInfo[2] & (1 << 19)) {
            defines << "#define SUPPORTS_SSE4_1\n";
        }
        if (cpuInfo[2] & (1 << 20)) {
            defines << "#define SUPPORTS_SSE4_2\n";
        }
        if (cpuInfo[2] & (1 << 28)) {
            defines << "#define SUPPORTS_AVX\n";
        }
        // AVX2 is bit 5 of the fourth element (index 3)
        if (cpuInfo[3] & (1 << 5)) {
            defines << "#define SUPPORTS_AVX2\n";
        }
        // Add more checks for other instruction sets as needed
    }
#else
    unsigned int eax, ebx, ecx, edx;

    // Get the highest function supported by CPUID
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);

    if (eax >= 1) {
        // Get the feature information
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);

        if (ecx & bit_SSE3) {
            defines << "#define SUPPORTS_SSE3\n";
        }
        if (ecx & bit_SSSE3) {
            defines << "#define SUPPORTS_SSSE3\n";
        }
        if (ecx & bit_SSE4_1) {
            defines << "#define SUPPORTS_SSE4_1\n";
        }
        if (ecx & bit_SSE4_2) {
            defines << "#define SUPPORTS_SSE4_2\n";
        }
        if (ecx & bit_AVX) {
            defines << "#define SUPPORTS_AVX\n";
        }
        if (ecx & bit_AVX2) {
            defines << "#define SUPPORTS_AVX2\n";
        }
        // Add more checks for other instruction sets as needed
    }
#endif

    return 0;
}
