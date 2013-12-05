extern "C" __global__ void test5 ( int* c , int* c_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 int x = 3 ;
 c [i] = ( c [i] ) + ( x ) ;
 } 