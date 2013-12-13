extern "C" __global__ void main0 ( int* a , int* a_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 int b [3] = {
 1 , 2 , 3 } ;
 a [i] = b [ 0 ] ;
 } 