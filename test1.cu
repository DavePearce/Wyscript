extern "C" __global__ void test1 ( int* l , int* l_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 l [i] = 0 ;
 } 