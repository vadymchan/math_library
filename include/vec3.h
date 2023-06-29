

class vec3
{



public:

	vec3()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	vec3(int x, int y, int z) : x(x), y(y), z(z)
	{
	}

	vec3 operator+(const vec3& other)
	{
		return vec3(x + other.x, y + other.y, z + other.z);
	}

	int x, y, z;

};