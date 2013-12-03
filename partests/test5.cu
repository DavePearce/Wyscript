__global__ void test2 ( int* c , int* c_length ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	int x = 3 ;
	c [i] = c [i] + x ;
}