__global__ void test3 ( int* l , int* l_length ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	l [i] = ( l [i] ) + ( 1 ) ;
}