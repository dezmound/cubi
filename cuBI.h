#ifndef CUBI_LIB
#define CUBI_LIB
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip> 
#include <sstream>

#ifndef __DEBUG
#define __DEBUG 1
#endif

#define CUBI_TRACE(message) if(__DEBUG)\
								std::cout << message << std::endl;
#define CU_HAS_ERRORS(A) if(A != cudaSuccess){\
 							printf("Error has in cuda function - %i at line: %i \n", A, __LINE__);\
 							exit(-1);}

/**
	Add fisrst vector with second
*/
__global__ void __cuAdd(unsigned char* a, unsigned char* b, unsigned char* c, unsigned char* overflowBytesList, long n);
/**
	Check what array contains only 0 values
*/
__device__ bool isEmptyArray(unsigned char* array, long length);

namespace cubi{
	class cuBI
	{
	private:
		unsigned char* bytes;
		long bytesCount;
		cuBI* add(cuBI* a, cuBI* b);
	public:
		cuBI(int a);
		~cuBI();
		cuBI(unsigned char* bytes, long bytesCount);
		cuBI operator +(cuBI b);
		std::string toHexString();
	};
}

#endif