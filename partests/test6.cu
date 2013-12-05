__global__ void test6 ( int* a , int* a_length , int* b , int* b_length , int* c , int* c_length ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x ;
	int x = ( a [i] ) + ( ( b [i] ) - ( ( a [i] ) * ( b [i] ) ) ) ;
	c [i] = ( ( x ) * ( x ) ) - ( ( x ) + ( 2 ) ) ;
}