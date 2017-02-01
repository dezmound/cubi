#include "cuBI.h"

using namespace cubi;

cuBI::cuBI(int a){
	this->bytesCount = sizeof(int) * sizeof(unsigned char);
	this->bytes = (unsigned char*)malloc(this->bytesCount);
	memcpy(this->bytes, (void*)&a, sizeof(int));
}

cuBI::cuBI(unsigned char* bytes, long bytesCount){
	this->bytesCount = bytesCount;
	this->bytes = (unsigned char*)malloc(this->bytesCount);
	memcpy(this->bytes, bytes, bytesCount);
}

cuBI::~cuBI(){
	free(this->bytes);
}

__device__ bool isEmptyArray(unsigned char* array, long length){
	for (long i = 0; i < length; i++)
	{
		if(array[i] > 0)
			return false;
	}
	return true;
}

__global__ void __cuAdd(unsigned char* a, unsigned char* b, unsigned char* c, unsigned char* overflowBytesList, long n){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int remainder = 0, divider = 0, sum = 0;
	while (tid < n){
		sum = (unsigned int)a[tid] + (unsigned int)b[tid];
		printf("%i + %i = %i\n", a[tid], b[tid], sum);
		divider = sum / 255;
		if(divider > 0) {
			remainder = sum % 255;
			overflowBytesList[tid + 1] = 1;
			c[tid] = (unsigned char)remainder;
		} else {
			c[tid] = (unsigned char)sum;
		}
		tid += blockDim.x * gridDim.x;
	}
	tid = threadIdx.x + blockIdx.x * blockDim.x;
	while(!isEmptyArray(overflowBytesList, n) && tid < n){
		sum = (unsigned int)c[tid] + overflowBytesList[tid];
		divider = sum / 255;
		if(divider > 0) {
			remainder = sum % 255;
			overflowBytesList[tid + 1] = 1;
			c[tid] = (unsigned char)remainder;
		} else {
			c[tid] = (unsigned char)sum;
		}
		tid += blockDim.x * gridDim.x;
	}
}

cuBI* cuBI::add(cuBI* a, cuBI* b){
	long minLength = min(a->bytesCount, b->bytesCount);
	long maxLength = max(a->bytesCount, b->bytesCount);
	unsigned char *overflowBytesList, *dev_a, *dev_b, *dev_c, *host_c;
	if(__DEBUG){
		CUBI_TRACE(std::string("A - 0x") + a->toHexString())
		CUBI_TRACE(std::string("B - 0x") + b->toHexString())
	}
	CUBI_TRACE("I am on add")
	CU_HAS_ERRORS(cudaMalloc((void**)&dev_a, a->bytesCount * sizeof(unsigned char)))
	CU_HAS_ERRORS(cudaMalloc((void**)&dev_b, b->bytesCount * sizeof(unsigned char)))
	CU_HAS_ERRORS(cudaMalloc((void**)&overflowBytesList, maxLength * sizeof(unsigned char)))
	CU_HAS_ERRORS(cudaMalloc((void**)&dev_c, maxLength * sizeof(unsigned char) + 1))
	CUBI_TRACE("I try copy memory to device")
	CU_HAS_ERRORS(cudaMemcpy(dev_a, a->bytes, a->bytesCount * sizeof(unsigned char), cudaMemcpyHostToDevice))
	CU_HAS_ERRORS(cudaMemcpy(dev_b, b->bytes, b->bytesCount * sizeof(unsigned char), cudaMemcpyHostToDevice))
	CUBI_TRACE("I try set memory of overflowBytesList")
	CU_HAS_ERRORS(cudaMemset(overflowBytesList, 0, maxLength * sizeof(unsigned char)))
	CU_HAS_ERRORS(cudaMemset(dev_c, 0, maxLength * sizeof(unsigned char) + 1))
	CUBI_TRACE("I starting add operation")
	__cuAdd<<<128,128>>>(dev_a, dev_b, dev_c, overflowBytesList, maxLength);
	cudaDeviceSynchronize();
	CUBI_TRACE("I complite add");
	CUBI_TRACE("I try free memory")
	host_c = (unsigned char*)malloc(maxLength * sizeof(unsigned char) + 1);
	CU_HAS_ERRORS(cudaMemcpy(host_c, dev_c, maxLength * sizeof(unsigned char) + 1, cudaMemcpyDeviceToHost))
	cuBI* result = new cuBI(host_c, maxLength * sizeof(unsigned char) + 1);
	CU_HAS_ERRORS(cudaFree(dev_a))
	CU_HAS_ERRORS(cudaFree(dev_b))
	CU_HAS_ERRORS(cudaFree(dev_c))
	CU_HAS_ERRORS(cudaFree(overflowBytesList))
	free(host_c);
	return result;
}

cuBI cuBI::operator +(cuBI b){
	return *this->add(this, &b);
}

std::string cuBI::toHexString(){
	std::stringstream _stringStream;
	for (int i = 0; i < this->bytesCount; i++)
	{
		_stringStream << std::setfill ('0') << std::setw(sizeof(unsigned char)*2)  << std::hex << (int)this->bytes[i];
	}
	return std::string(_stringStream.str());
}