#include <vec3.h>

#include <iostream>

int main()
{
	vec3 v1(1, 2, 3);
	vec3 v2(4, 5, 6);
	vec3 v3 = v1 + v2;

	std::cout << v3.x << ", " << v3.y << ", " << v3.z << std::endl;

	return 0;
}
