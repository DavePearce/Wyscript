extern "C" __global__ void main0 ( int* a , int* a_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 a [i] = 0 ;
 } 