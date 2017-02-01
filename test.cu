#define __DEBUG 1
#include "cuBI.h"
#include <iostream>

using namespace cubi;

int main(){
	cuBI a(255), b(255), c(0);
	std::cout << "Begin of the program" << std::endl;
	std::cout << (a + b).toHexString() << " print" << std::endl;
	return 0;
}