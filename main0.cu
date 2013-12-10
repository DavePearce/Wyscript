extern "C" __global__ void main0 ( int* a , int* a_length , int* b , int* b_length ) {
 int i = blockIdx.x * blockDim.x + threadIdx.x ;
 if ( i < ( * a_length ) && i >= 0 ) {
 b [i] = a [i] ;
 } } 