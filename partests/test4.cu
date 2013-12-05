__global__ void test4 ( int* c , int* c_length , int* a , int* a_length , int* b , int* b_length ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	c [i] = ( ( a [i] ) * ( b [i] ) ) + ( ( 3 ) - ( a [i] ) ) ;
}